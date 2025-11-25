import os
from urllib.parse import quote_plus
from typing import Literal, List

from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
from backend.core.config import settings
from backend.utils.custom_exception import CustomException
from backend.utils.logger import get_logger
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver # <--- NEW IMPORT

# Import new modules
from backend.src.custom_tools import get_db_tools
from backend.src.prompt_module import (
    select_table_prompt_module, 
    generate_query_prompt_module, 
    query_verification_prompt_module,
    answer_validation_prompt_module
)

logger = get_logger(__name__)

class SQLAgentGenerator:
    def __init__(self, api_key: str = None, model_name: str = "google_genai:gemini-2.5-flash-lite"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        
        if self.api_key:
            os.environ["GOOGLE_API_KEY"] = self.api_key
        
        self.llm = self._setup_llm()
        self.db = self._setup_database()

        self.tools = self._setup_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        # Initialize Memory Persistence
        self.checkpointer = MemorySaver() # <--- NEW: In-memory persistence
        
        self.graph = self._build_graph()
        
        logger.info("SQL Agent Initialized with Memory")

    def _setup_llm(self):
        return init_chat_model(self.model_name)

    def _setup_database(self) -> SQLDatabase:
        try:
            encoded_user = quote_plus(settings.DB_USER)
            encoded_password = quote_plus(settings.DB_PASSWORD)
            encoded_name = quote_plus(settings.DB_NAME)
            db_uri = f"mysql+pymysql://{encoded_user}:{encoded_password}@{settings.DB_HOST}/{encoded_name}"
            return SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=0)
        except Exception as e:
            logger.error("Error in setting up database")
            raise CustomException("Error in setting up database", e)
    
    def _setup_tools(self) -> List:
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        standard_tools = toolkit.get_tools()
        custom_tools = get_db_tools(self.db)
        return standard_tools + custom_tools

    # --- Nodes ---

    def list_tables_node(self, state: MessagesState):
        # Optimization: Only list tables if this is the FIRST message in history
        # Otherwise, the agent remembers tables from previous turns.
        if len(state["messages"]) > 2:
             return {"messages": []}

        tool_call = {"name": "sql_db_list_tables", "args": {}, "id": "init_list_tables", "type": "tool_call"}
        tool_call_message = AIMessage(content="", tool_calls=[tool_call])
        
        list_tables_tool = self.tool_map["sql_db_list_tables"]
        tool_output = list_tables_tool.invoke({}) 
        
        response = AIMessage(content=f"Available tables: {tool_output}")
        return {"messages": [tool_call_message, response]}

    def call_get_schema_node(self, state: MessagesState):
        system_message = {"role": "system", "content": select_table_prompt_module()}
        llm_with_tools = self.llm.bind_tools([self.tool_map["sql_db_schema"]], tool_choice="any")
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    def generate_query_node(self, state: MessagesState):
        system_message = {"role": "system", "content": generate_query_prompt_module(self.db)}
        tools_to_bind = [
            self.tool_map["sql_db_query"], 
            self.tool_map["sql_db_query_distinct_values"],
            self.tool_map["sql_db_sample_rows"]
        ]
        llm_with_tools = self.llm.bind_tools(tools_to_bind)
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    def check_query_node(self, state: MessagesState):
        """Verifies query and fixes 'Raw SQL' hallucinations."""
        last_message = state["messages"][-1]
        
        if not last_message.tool_calls:
            content = last_message.content.strip()
            if content.upper().startswith("SELECT"):
                logger.info("Detected raw SQL text. Converting to tool_call...")
                manual_tool_call = {
                    "id": "manual_sql_fix_" + os.urandom(4).hex(),
                    "name": "sql_db_query",
                    "args": {"query": content},
                    "type": "tool_call"
                }
                fixed_message = AIMessage(content="", tool_calls=[manual_tool_call])
                return {"messages": [fixed_message]}
            return {"messages": []}

        tool_call = last_message.tool_calls[0]
        
        if tool_call["name"] in ["sql_db_query_distinct_values", "sql_db_sample_rows"]:
                return {"messages": []} 

        if tool_call["name"] == "sql_db_query":
            system_message = {"role": "system", "content": query_verification_prompt_module(self.db)}
            proposed_query = tool_call["args"].get("query")
            user_message = {"role": "user", "content": f"Verify this query: {proposed_query}"}
            
            run_query_tool = self.tool_map["sql_db_query"]
            llm_with_tools = self.llm.bind_tools([run_query_tool], tool_choice="any")
            
            response = llm_with_tools.invoke([system_message, user_message])
            return {"messages": [response]}
        
        return {"messages": []}

    def validate_answer_node(self, state: MessagesState):
        last_message = state["messages"][-1]
        sql_result = last_message.content
        
        # Get the MOST RECENT user question (not the very first one in history)
        # We search backwards for the last HumanMessage
        user_question = "Unknown"
        for msg in reversed(state["messages"]):
             if isinstance(msg, HumanMessage):
                  user_question = msg.content
                  break

        generated_query = "Unknown"
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                if msg.tool_calls[0]["name"] == "sql_db_query":
                    generated_query = msg.tool_calls[0]["args"].get("query")
                    break

        prompt = answer_validation_prompt_module().format(
            question=user_question, 
            query=generated_query, 
            result=sql_result
        )
        
        validation_response = self.llm.invoke(prompt)
        
        if "STATUS: RETRY" in validation_response.content:
            logger.info("Validator Triggered Retry")
            feedback_msg = HumanMessage(content=f"Previous query returned unsatisfactory results. Validator Feedback: {validation_response.content}. Please try a new approach (check values/samples if needed).")
            return {"messages": [feedback_msg]}
            
        return {"messages": [validation_response]}

    def generate_final_answer_node(self, state: MessagesState):
        # Get latest User Question
        user_question = "Unknown"
        for msg in reversed(state["messages"]):
             if isinstance(msg, HumanMessage):
                  user_question = msg.content
                  break
        
        sql_result = "No data found."
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage) and msg.name == "sql_db_query":
                sql_result = msg.content
                break
                
        prompt = f"""
        User Question: {user_question}
        SQL Result: {sql_result}
        
        Please provide a concise, natural language answer to the user's question based on the SQL Result. 
        If the result is a list, summarize it nicely. Do not mention SQL code.
        """
        final_response = self.llm.invoke(prompt)
        return {"messages": [final_response]}

    # --- Edges ---

    def should_continue(self, state: MessagesState) -> str:
        last_message = state["messages"][-1]
        if not last_message.tool_calls:
            return "end"
        return "check_query"

    def should_retry(self, state: MessagesState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage) and "Validator Feedback" in last_message.content:
            return "generate_query"
        return "generate_final_answer"

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(MessagesState)
        tools_node = ToolNode(self.tools)

        workflow.add_node("list_tables", self.list_tables_node)
        workflow.add_node("call_get_schema", self.call_get_schema_node)
        workflow.add_node("get_schema", tools_node) 
        workflow.add_node("generate_query", self.generate_query_node)
        workflow.add_node("check_query", self.check_query_node)
        workflow.add_node("run_tools", tools_node) 
        workflow.add_node("validate_answer", self.validate_answer_node)
        workflow.add_node("generate_final_answer", self.generate_final_answer_node)

        workflow.add_edge(START, "list_tables")
        workflow.add_edge("list_tables", "call_get_schema")
        workflow.add_edge("call_get_schema", "get_schema")
        workflow.add_edge("get_schema", "generate_query")
        
        workflow.add_conditional_edges("generate_query", self.should_continue, {"check_query": "check_query", "end": END})
        workflow.add_edge("check_query", "run_tools")
        workflow.add_edge("run_tools", "validate_answer")
        workflow.add_conditional_edges("validate_answer", self.should_retry, {"generate_query": "generate_query", "generate_final_answer": "generate_final_answer"})
        workflow.add_edge("generate_final_answer", END)

        # COMPILE WITH CHECKPOINTER
        return workflow.compile(checkpointer=self.checkpointer)

    def run(self, question: str, session_id: str = "default_session", config: RunnableConfig = None):
        """
        Executes the agent with memory enabled via session_id.
        """
        # Configure the thread_id for LangGraph persistence
        config = config or {}
        config["configurable"] = {"thread_id": session_id}
        
        initial_state = {"messages": [{"role": "user", "content": question}]}
        
        final_response_content = ""
        
        # Pass the config to stream so it knows which thread to append to
        for step in self.graph.stream(initial_state, config=config, stream_mode="values"):
            last_msg = step["messages"][-1]
            last_msg.pretty_print()
            
            if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                final_response_content = last_msg.content
            
        return final_response_content