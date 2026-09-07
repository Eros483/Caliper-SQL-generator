from typing import List

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from backend.core.planner import Plan, PlanStep, planner_system_prompt
from backend.core.tiered_strategy import build_enriched_prompt
from backend.src.custom_tools import analyze_data, generate_chart, get_db_tools, run_python_code_in_sandbox
from backend.src.graph_manager import SchemaGraph
from backend.src.prompt_module import (
    answer_validation_prompt_module,
    generate_query_prompt_module,
)
from backend.src.rag_manager import SchemaRAG
from backend.utils.config import settings
from backend.utils.custom_exception import CustomException
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class AgentState(MessagesState):
    plan: Plan | None
    user_context: dict
    enriched_prompt: str


class SQLAgentGenerator:
    def __init__(self, model_name: str = "google_genai:gemini-3.5-flash"):
        self.model_name = model_name

        if not settings.gemini_api_key:
            raise CustomException("GEMINI_API_KEY is not set in environment")

        self.llm = self._setup_llm()
        self.db = self._setup_database()
        self.rag = SchemaRAG(self.db)
        self.graph_manager = SchemaGraph(self.db)
        self.tools = self._setup_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

        logger.info(f"SQL Agent Initialized with model: {self.model_name}")

    def _setup_llm(self):
        logger.info(f"Initializing Gemini model: {self.model_name}")
        return init_chat_model(self.model_name, temperature=0)

    def _setup_database(self) -> SQLDatabase:
        try:
            return SQLDatabase.from_uri(settings.database_uri, sample_rows_in_table_info=0)
        except Exception as e:
            logger.error("Error setting up database")
            raise CustomException("Error setting up database", e)

    def _setup_tools(self) -> List:
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        standard_tools = toolkit.get_tools()
        custom_tools = get_db_tools(self.db, self.rag, self.graph_manager)
        return standard_tools + custom_tools + [run_python_code_in_sandbox, analyze_data, generate_chart]

    def _preprocess(self, question: str) -> str:
        from backend.core.ingestion import classify, normalize

        clean = normalize(question)
        label = classify(clean)
        if label != "benign":
            logger.warning(f"Ingestion classified as {label}: {question[:80]}")
            # hardened prompt — no schema internals
            return f"User Question: {clean}\n[Hardened mode: query-only tools, no schema internals exposed]"
        rag_result = self.rag.search_tables(clean)
        return build_enriched_prompt(clean, rag_result)

    def planner_node(self, state: AgentState) -> dict:
        base_prompt = state.get("enriched_prompt", "")
        # lever 1 — planner skip for Tier 1/2 (deterministic via enriched prompt, no LLM call)
        if "STRATEGY: Query only" in base_prompt or "STRATEGY: Join tables" in base_prompt:
            user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            last_question = user_messages[-1].content if user_messages else ""
            plan = Plan(steps=[PlanStep(step=1, action="query", description=last_question)])
            return {"plan": plan, "messages": []}

        system_message = {"role": "system", "content": planner_system_prompt()}

        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        last_question = user_messages[-1].content if user_messages else ""

        plan_prompt = f"""{base_prompt}

Decompose this question into a plan:
{last_question}"""

        response = self.llm.invoke([system_message, {"role": "user", "content": plan_prompt}])

        try:
            import json

            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            plan_data = json.loads(content)
            plan = Plan(**plan_data)
            logger.info(f"Plan created: {[s.description for s in plan.steps]}")
        except Exception as e:
            logger.warning(f"Plan parsing failed: {e}, using default plan")
            plan = Plan(steps=[PlanStep(step=1, action="query", description=last_question)])

        return {"plan": plan, "messages": [response]}

    def generate_query_node(self, state: AgentState) -> dict:
        org_id = state.get("user_context", {}).get("org_id", 16)
        plan = state.get("plan")
        base_prompt = state.get("enriched_prompt", generate_query_prompt_module(self.db, org_id=org_id))

        plan_step_text = ""
        if plan and plan.steps:
            pending = [s for s in plan.steps if s.status == "pending"]
            if pending:
                current = pending[0]
                current.status = "running"
                plan_step_text = f"\nCURRENT STEP: {current.description}"

        instruction = f"""
{plan_step_text}

### EXECUTION PLAN
1. **RESEARCH PHASE:** If you don't know table names, use `sql_db_find_relevant_tables`.
2. **CONNECTION PHASE:** If you don't know how to join tables, use `sql_db_find_table_connections`.
3. **EXECUTION PHASE (CRITICAL):** Once you have the table names and join logic, you **MUST** run `sql_db_query`.
4. **SECURITY:** You MUST filter by the Organization ID provided in the system prompt.

### DATA RULES
- Use `LIMIT 10` for all queries.
- For Binary IDs (like patient_id), use `BIN_TO_UUID(col)` or `HEX(col)`.
"""

        system_message = {"role": "system", "content": base_prompt + instruction}

        tools_to_bind = [
            self.tool_map["sql_db_query"],
            self.tool_map["sql_db_query_distinct_values"],
            self.tool_map["sql_db_sample_rows"],
            self.tool_map["sql_db_find_relevant_tables"],
            self.tool_map["sql_db_find_table_connections"],
            self.tool_map["sql_db_get_foreign_keys"],
            self.tool_map["sql_db_get_column_info"],
            self.tool_map["sql_db_find_value_location"],
        ]
        llm_with_tools = self.llm.bind_tools(tools_to_bind)
        response = llm_with_tools.invoke([system_message] + state["messages"])

        return {"messages": [response], "plan": plan}

    def check_query_node(self, state: AgentState) -> dict:
        from backend.core.guardrails import GuardrailError, guard_query

        org_id = state.get("user_context", {}).get("org_id")
        # local fallback keeps existing demo flow; prod will have real claim
        if org_id is None:
            org_id = 16

        last_message = state["messages"][-1]
        if not last_message.tool_calls:
            content = last_message.content.strip()
            if any(content.upper().startswith(kw) for kw in ["SELECT", "INSERT", "UPDATE", "DELETE"]):
                logger.warning("Detected raw SQL text. Converting to tool_call...")
                try:
                    guarded = guard_query(content, org_id)
                    content = guarded
                except GuardrailError as e:
                    # fail-closed — no retry, surface error directly
                    return {"messages": [AIMessage(content=f"Guardrail rejected query: {e}")]}
                import os as _os

                manual_tool_call = {
                    "id": "manual_sql_fix_" + _os.urandom(4).hex(),
                    "name": "sql_db_query",
                    "args": {"query": content},
                    "type": "tool_call",
                }
                return {"messages": [AIMessage(content="", tool_calls=[manual_tool_call])]}
            return {"messages": []}

        tool_call = last_message.tool_calls[0]
        tool_name = tool_call["name"]

        REASONING_TOOLS = [
            "sql_db_query_distinct_values",
            "sql_db_sample_rows",
            "sql_db_find_relevant_tables",
            "sql_db_schema",
            "sql_db_get_foreign_keys",
            "sql_db_get_column_info",
            "sql_db_find_table_connections",
        ]
        if tool_name in REASONING_TOOLS:
            return {"messages": []}

        if tool_name == "sql_db_query":
            proposed_query = tool_call["args"].get("query", "")
            try:
                guarded = guard_query(proposed_query, org_id)
            except GuardrailError as e:
                return {"messages": [AIMessage(content=f"Guardrail rejected query: {e}")]}
            if guarded != proposed_query:
                return {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": tool_call["id"],
                                    "name": "sql_db_query",
                                    "args": {"query": guarded},
                                    "type": "tool_call",
                                }
                            ],
                        )
                    ]
                }

        return {"messages": []}

    def validate_answer_node(self, state: AgentState) -> dict:
        last_message = state["messages"][-1]

        if isinstance(last_message, AIMessage) and not last_message.tool_calls:
            content_lower = last_message.content.lower()

            premature_exit_phrases = [
                "no matching records",
                "no data was found",
                "no medicaid patients",
                "no patients",
                "could not find",
            ]

            if any(phrase in content_lower for phrase in premature_exit_phrases):
                user_question = "Unknown"
                for msg in reversed(state["messages"]):
                    if isinstance(msg, HumanMessage) and not msg.content.startswith("SYSTEM"):
                        user_question = msg.content
                        break

                recent_tool_calls = []
                for msg in reversed(state["messages"][-10:]):
                    if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                        recent_tool_calls.extend([tc["name"] for tc in msg.tool_calls])

                if "sql_db_query" not in recent_tool_calls:
                    logger.warning("Agent terminated without executing sql_db_query. Forcing retry.")
                    return {
                        "messages": [
                            HumanMessage(
                                content=f"""SYSTEM FEEDBACK: You aborted without running a SELECT query.

    You used helper tools ({", ".join(set(recent_tool_calls))}) but never executed the actual data retrieval.

    MANDATORY NEXT STEP: Use sql_db_query to run the following query structure:

    For '{user_question}', build a query that:
    1. JOINs the necessary tables
    2. Filters WHERE conditions match the user's criteria
    3. Orders and limits results appropriately
    4. Uses HEX() or BIN_TO_UUID() for any BINARY columns

    Execute the query NOW with sql_db_query."""
                            )
                        ]
                    }

        if isinstance(last_message, ToolMessage):
            if len(state["messages"]) >= 2:
                last_ai_msg = None
                for msg in reversed(state["messages"][:-1]):
                    if isinstance(msg, AIMessage):
                        last_ai_msg = msg
                        break

                if last_ai_msg and hasattr(last_ai_msg, "tool_calls") and last_ai_msg.tool_calls:
                    tool_name = last_ai_msg.tool_calls[0]["name"]
                    HELPER_TOOLS = [
                        "sql_db_find_relevant_tables",
                        "sql_db_find_table_connections",
                        "sql_db_schema",
                        "sql_db_get_foreign_keys",
                        "sql_db_get_column_info",
                        "sql_db_query_distinct_values",
                        "sql_db_sample_rows",
                    ]

                    if tool_name in HELPER_TOOLS:
                        return {
                            "messages": [
                                HumanMessage(
                                    content=f"SYSTEM FEEDBACK: Research tool '{tool_name}' complete. Now write and execute 'sql_db_query'."
                                )
                            ]
                        }

        if not isinstance(last_message, ToolMessage) or last_message.name != "sql_db_query":
            return {"messages": []}

        sql_result = last_message.content
        plan = state.get("plan")
        if plan and plan.steps:
            for step in plan.steps:
                if step.status == "running":
                    step.status = "done"

        if "b'\\" in sql_result or "bytearray" in str(sql_result).lower():
            logger.warning("Binary data detected. Triggering retry.")
            return {
                "messages": [
                    HumanMessage(
                        content="SYSTEM FEEDBACK: Binary data detected. Retry using HEX(column) or BIN_TO_UUID(column)."
                    )
                ]
            }

        # lever 3 — deterministic validation on happy path: rows returned + no binary => skip LLM validate
        if (
            sql_result
            and sql_result.strip()
            and "no rows" not in sql_result.lower()
            and "error" not in sql_result.lower()
        ):
            # sqlglot already validated the query via check_query_node; skip LLM call
            return {"messages": [AIMessage(content="STATUS: VALID")], "plan": plan}

        user_question = "Unknown"
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage) and not msg.content.startswith("SYSTEM"):
                user_question = msg.content
                break

        generated_query = "Unknown"
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                if msg.tool_calls[0]["name"] == "sql_db_query":
                    generated_query = msg.tool_calls[0]["args"].get("query")
                    break

        validation_prompt = answer_validation_prompt_module().format(
            question=user_question, query=generated_query, result=sql_result
        )

        validation_response = self.llm.invoke(validation_prompt)

        if "STATUS: RETRY" in validation_response.content:
            logger.info("Validator Triggered Retry")
            retry_count = sum(
                1 for msg in state["messages"] if isinstance(msg, HumanMessage) and "FEEDBACK" in msg.content
            )

            if retry_count >= 3:
                return {"messages": [AIMessage(content="Maximum retries reached. Answering with available data.")]}

            feedback_text = validation_response.content.split("FEEDBACK:")[-1].strip()
            return {"messages": [HumanMessage(content=f"Validator Feedback: {feedback_text}")]}

        return {"messages": [validation_response], "plan": plan}

    def generate_final_answer_node(self, state: AgentState) -> dict:
        user_question = "Unknown"
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage) and not msg.content.startswith("SYSTEM"):
                user_question = msg.content
                break

        sql_result = "No data found."
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage) and msg.name == "sql_db_query":
                sql_result = msg.content
                break

        # trust: frame rows as untrusted tool content with nonce (persists in memory)
        from backend.core.ingestion import coverage_check, wrap_untrusted

        nonce = state.get("user_context", {}).get("nonce", "trust")
        framed = wrap_untrusted(str(sql_result), nonce=nonce)

        prompt = f"""
        User Question: {user_question}
        SQL Result (untrusted tool output): {framed}

        Provide a concise, natural language answer.
        - If the result is a list, summarize it.
        - If the result is empty, explain that no matching records were found.
        """
        final_response = self.llm.invoke(prompt)
        # §7 coverage check — every aggregate must appear in answer
        if not coverage_check(final_response.content, str(sql_result)):
            logger.warning("Coverage check failed — answer may omit aggregates")
        return {"messages": [final_response]}

    def should_continue(self, state: AgentState) -> str:
        last_message = state["messages"][-1]
        if not last_message.tool_calls:
            return "end"
        return "check_query"

    def should_retry(self, state: AgentState) -> str:
        last_message = state["messages"][-1]

        if isinstance(last_message, HumanMessage) and (
            "FEEDBACK" in last_message.content or "Validator" in last_message.content
        ):
            return "generate_query"

        if isinstance(last_message, AIMessage) and "STATUS: VALID" in last_message.content:
            return "generate_final_answer"

        return "generate_final_answer"

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        tools_node = ToolNode(self.tools)

        workflow.add_node("planner", self.planner_node)
        workflow.add_node("generate_query", self.generate_query_node)
        workflow.add_node("check_query", self.check_query_node)
        workflow.add_node("run_tools", tools_node)
        workflow.add_node("validate_answer", self.validate_answer_node)
        workflow.add_node("generate_final_answer", self.generate_final_answer_node)

        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "generate_query")

        workflow.add_conditional_edges(
            "generate_query",
            self.should_continue,
            {"check_query": "check_query", "end": END},
        )

        workflow.add_edge("check_query", "run_tools")
        workflow.add_edge("run_tools", "validate_answer")

        workflow.add_conditional_edges(
            "validate_answer",
            self.should_retry,
            {
                "generate_query": "generate_query",
                "generate_final_answer": "generate_final_answer",
            },
        )

        workflow.add_edge("generate_final_answer", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def _stream_trace(self, initial_state: dict, config: dict) -> dict:
        final_response = ""
        sql_queries: list[str] = []
        sql_results: list[str] = []
        retries = 0

        try:
            for step in self.graph.stream(initial_state, config=config, stream_mode="values"):
                last_msg = step["messages"][-1]

                if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        if tc.get("name") == "sql_db_query":
                            query = tc.get("args", {}).get("query", "")
                            if query:
                                sql_queries.append(query)

                if isinstance(last_msg, ToolMessage) and last_msg.name == "sql_db_query":
                    sql_results.append(str(last_msg.content))

                if isinstance(last_msg, HumanMessage) and "FEEDBACK" in last_msg.content:
                    retries += 1

                if isinstance(last_msg, AIMessage) and not last_msg.tool_calls and "STATUS:" not in last_msg.content:
                    final_response = last_msg.content
        except Exception as e:
            logger.error(f"Error streaming: {e}")
            final_response = f"Error: {e}"

        return {
            "response": final_response,
            "sql_queries": sql_queries,
            "sql_results": sql_results,
            "retries": retries,
        }

    def run(
        self,
        question: str,
        session_id: str = "default_session",
        config: RunnableConfig = None,
        org_id: int | None = None,
    ) -> str:
        config = config or {}
        config["configurable"] = {"thread_id": session_id}
        config["recursion_limit"] = 50

        enriched_prompt = self._preprocess(question)

        initial_state = {
            "messages": [SystemMessage(content=enriched_prompt), HumanMessage(content=question)],
            "user_context": {"org_id": org_id},
            "enriched_prompt": enriched_prompt,
            "plan": None,
        }

        logger.info(f"Session: {session_id} | Query: {question}")
        trace = self._stream_trace(initial_state, config)
        return trace["response"]

    def run_with_trace(
        self,
        question: str,
        session_id: str = "eval_session",
        org_id: int | None = None,
    ) -> dict:
        import uuid

        from backend.core.audit import audit_log
        from backend.core.semantic_cache import semantic_cache

        # lever 5 — semantic cache hit = skip generation
        cache_k = semantic_cache.make_key(question, org_id)
        cached_sql = semantic_cache.get(cache_k)
        if cached_sql:
            try:
                from backend.core.db import run_org_scoped

                rows = run_org_scoped(self.db, cached_sql, org_id or 16)
                answer = self.llm.invoke(
                    f"User Question: {question}\nSQL Result: {rows}\nProvide concise answer."
                ).content
                return {"response": answer, "sql_queries": [cached_sql], "sql_results": [rows], "retries": 0}
            except Exception:
                pass  # fall through to normal path

        config = {"configurable": {"thread_id": session_id}, "recursion_limit": 50}

        enriched_prompt = self._preprocess(question)
        nonce = uuid.uuid4().hex[:8]

        initial_state = {
            "messages": [SystemMessage(content=enriched_prompt), HumanMessage(content=question)],
            "user_context": {"org_id": org_id, "nonce": nonce},
            "enriched_prompt": enriched_prompt,
            "plan": None,
        }

        logger.info(f"Eval Session: {session_id} | Query: {question}")
        trace = self._stream_trace(initial_state, config)
        # store successful query in cache
        if trace["sql_queries"]:
            semantic_cache.set(cache_k, trace["sql_queries"][-1])
        # compliance plane — append-only audit
        try:
            audit_log(
                question,
                trace["sql_queries"][-1] if trace["sql_queries"] else "",
                str(trace["sql_results"]),
                org_id,
                trace_id=session_id,
            )
        except Exception:
            pass
        return trace
