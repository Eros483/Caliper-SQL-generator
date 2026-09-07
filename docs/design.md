# CaliperLens — Design & Architecture

## Overview

CaliperLens is an agentic natural-language-to-SQL engine for healthcare datasets. A user types a plain-English question ("top 5 Medicaid patients by highest SDOH score, chart their costs") and the system autonomously discovers relevant tables via semantic search, generates validated SQL, executes it inside a sandbox, optionally performs statistical analysis and produces charts, and returns a natural-language answer with embedded visuals.

The stack runs entirely locally (Docker). Only the frontend is deployed to Vercel as a static shell with a HIPAA compliance notice. Patient data never leaves the developer's machine.

---

## Architecture

![CaliperLens Local Architecture](images/local_architecture.png)

*Generated via [mingrammer/diagrams](https://github.com/mingrammer/diagrams) — `make diagram` — source: `docs/diagram.py`*

```
┌───────────────────────────────────────────────────────────────────┐
│                        Local Docker Host                           │
│                                                                   │
│  ┌──────────┐   ┌──────────┐   ┌───────────────┐                  │
│  │ FastAPI  │   │ Airflow  │   │ Prometheus    │                  │
│  │ Backend  │   │ (dbt)    │   │ + Grafana     │                  │
│  │ :8000    │   │          │   │ :9090 :3001   │                  │
│  └─────┬────┘   └────┬─────┘   └───────────────┘                  │
│        │             │                                            │
│  ┌─────▼──────────┐  │  ┌──────────────────┐                      │
│  │ LangGraph Agent│  │  │ dbt transforms   │                      │
│  │ Gemini 3.5     │  │  │ MySQL → DuckDB   │                      │
│  │ Flash          │  │  │ staging → inter  │                      │
│  │                │  │  │ mediate → marts  │                      │
│  └───┬───┬───┬────┘  │  └──────────────────┘                      │
│      │   │   │       │                                            │
│  ┌───▼─┐ ┌▼──▼──┐ ┌─▼──────────┐                                  │
│  │RAG  │ │Graph │ │  Sandbox    │                                  │
│  │FAISS│ │NetwkX│ │  Docker     │                                  │
│  │     │ │      │ │  --net=none │                                  │
│  └─────┘ └──────┘ │  ro mounts  │                                  │
│                   └─────────────┘                                  │
└───────────────────────────────────────────────────────────────────┘
```

### Data flow

1. **Ingestion**: MySQL dump → dbt pipeline (staging → intermediate → marts) → DuckDB analytics file
2. **Preprocessing**: User question → FAISS RAG search → enriched prompt with relevant tables + tier strategy
3. **Agent execution**: 6-node LangGraph (planner → generate → check → execute → validate → answer) with retry loop
4. **Analysis/Charts**: Agent writes Python → sandbox Docker container executes → results returned
5. **Observability**: Prometheus `/metrics` + LangSmith traces + JSON structured logs with trace IDs

---

## Technology Choices

| Choice | Why |
|---|---|
| **DuckDB** | In-process OLAP, zero servers, single-file database. Blazing fast analytics. |
| **dbt (dbt-duckdb)** | Industry-standard analytics engineering. Pre-compute complex joins at build time instead of discovering them at query time. |
| **Apache Airflow** | Schedules daily dbt runs. Dockerized, single-node. |
| **Gemini 3.5 Flash** | Free tier, strong SQL generation. Same provider for embeddings — single API key. |
| **LangGraph** | State-machine agent orchestration with built-in checkpointing, tool binding, and conditional edges. |
| **FAISS + text-embedding-004** | Semantic schema search. Maps user terminology ("Medicaid") to technical table names. |
| **NetworkX (SchemaGraph)** | Dijkstra join-pathfinding for edge-case queries the data marts cannot answer. |
| **Docker Sandbox** | All generated code runs isolated: no network, memory/cpu caps, timeout, read-only DB. |
| **JWT (python-jose + passlib bcrypt)** | Access + refresh tokens, rate-limited per user. org_id scoping from claims. |
| **Prometheus + Grafana** | Request latency, error rate, throughput metrics. Pre-built dashboard. |
| **Structured JSON logging** | Every log line carries trace_id + session_id + node for end-to-end reconstruction. |
| **Zustand** | 1KB state management for frontend chat state. |
| **BUSL-1.1** | Free for personal/portfolio/educational use. Commercial use requires license. |

---

## The Data Pipeline

### Why it exists

Raw MySQL tables are normalized (3NF). A query like "top Medicaid patients by SDOH score" requires a 4-hop join: `patient → map_patient_metrics → lob → patient_score` through a bridge table. The agent used NetworkX/Dijkstra to discover this path every time.

dbt pre-computes these joins at build time. The agent queries a single denormalized mart (`fct_patient_metrics`) that already contains demographics, insurance, and all five risk scores in one row. The join complexity moved from query-time (where the LLM stumbles) to build-time (where it is deterministic and testable).

### dbt layers

**Staging (10 models, views):** 1:1 mirrors of MySQL tables. `LOAD mysql; ATTACH ... AS mysql_source (TYPE mysql, READ_ONLY)` connects DuckDB to MySQL at the start of each run. Views — DuckDB reads through transparently, no data copy.

**Intermediate (4 models, tables):** Pre-computed join paths. `int_patient_insurance` resolves the 4-hop `patient → map_patient_metrics → lob → organization` path. `int_patient_scores` keeps only the latest score per patient. `int_patient_conditions` and `int_patient_interventions` resolve the diagnosis and intervention paths. Materialized as tables so the join is done once, not per query.

**Marts (4 models, tables):** Analytics-ready, denormalized. `fct_patient_metrics` is the workhorse — one row per patient with everything pre-joined. `fct_interventions`, `dim_patients`, `dim_conditions` cover the rest. All marts share `patient_id` as a common key.

Every mart has `not_null` and `unique` tests on primary keys, and `relationships` tests across marts. Model descriptions with medical terminology feed into the RAG index.

### Airflow orchestration

Runs locally via `docker-compose.airflow.yaml` (webserver + scheduler + postgres). The `caliperlens_pipeline` DAG runs daily: `dbt deps → dbt run → dbt test → dbt docs`. Manual trigger via `make dbt-run`.

---

## Tiered Query Strategy

Not every question needs full analytical firepower. The system classifies every query into one of three tiers in `backend/core/tiered_strategy.py`:

**Tier 1 — Single mart (~80% of queries):** RAG returns tables from one mart. Zero joins. `SELECT ... FROM fct_patient_metrics WHERE ...`.

**Tier 2 — Two-mart join (~15%):** RAG returns tables from two marts. Both share `patient_id`. Simple `JOIN ON patient_id = patient_id`. The FK dictionary (`MART_SHARED_KEYS`) confirms shared keys; no pathfinding needed.

**Tier 3 — Staging fallback (~5%):** The marts do not cover this query. Agent falls back to staging views (1:1 MySQL mirrors) and uses the SchemaGraph (NetworkX/Dijkstra) to discover join paths. The graph is preserved as a safety net.

---

## The Agent

### Preprocessing

Before the LangGraph runs, the user's question is passed through the FAISS RAG. The vector store was indexed at startup against every database table, enriched with hand-written business context (table roles, medical terminology, query patterns). The search returns top-K semantically relevant tables with their schemas.

The `tiered_strategy` module classifies the result and builds an enriched prompt containing: user question, relevant table schemas, and an explicit strategy instruction ("No joins needed" or "Join on patient_id").

### Graph

6 nodes, 2 conditional edges, 1 retry loop:

```
planner → generate → check → run_tools → validate → final_answer → END
             ↑__________________retry_________________↓
```

**Planner:** Decomposes the question into a structured `Plan` (Pydantic-validated JSON with `PlanStep` items: step number, action type `query|analyze|chart`, description, status). On failure, a `re_plan_prompt` generates a revised approach.

**Generate Query:** Binds 12 tools to the LLM (SQL execution, RAG search, graph pathfinding, schema inspection, data sampling, Python sandbox execution, analysis, chart generation). System prompt includes the enriched RAG context plus a mandatory instruction: *do not stop until you have run a SELECT query and received data rows.*

**Check Query:** Syntax validator. Converts raw SQL text into tool calls. Enforces `LIMIT 10`. Passes reasoning/research tools through unchanged.

**Run Tools:** LangGraph `ToolNode` executes whatever the LLM requested.

**Validate Answer:** Four-layer quality assurance. Detects premature termination (gave up before querying). Forces progression from research tools to execution. Detects binary garbage (forces `HEX()` retry). Sends SQL + result to Gemini for final semantic validation. Retries up to 3 times before answering with available data.

**Generate Final Answer:** Synthesizes SQL results into natural language.

### Tools exposed to the LLM

| Tool | Type | Responsibility |
|---|---|---|
| `sql_db_query` | Execution | Execute SQL against the database |
| `sql_db_find_relevant_tables` | Research (RAG) | Semantic vector search for table discovery |
| `sql_db_find_table_connections` | Reasoning (Graph) | Dijkstra-based join pathfinding |
| `sql_db_query_distinct_values` | Research | Inspect unique values in a column (with keyword filter) |
| `sql_db_sample_rows` | Research | Preview 3 rows from a table |
| `sql_db_get_foreign_keys` | Reasoning | List explicitly defined foreign keys |
| `sql_db_get_column_info` | Research | Detailed column type/comment inspection |
| `sql_db_find_value_location` | Research | Locate which table/column holds a given value |
| `run_python_code_in_sandbox` | Execution (Sandbox) | Arbitrary Python in an isolated Docker container |
| `analyze_data` | Execution (Sandbox) | Statistics (mean/median/stddev, trend detection) |
| `generate_chart` | Execution (Sandbox) | matplotlib charts (bar/line/scatter/pie/histogram) |

Only the first eight are bound during SQL generation (`generate_query`); the sandbox tools are bound for the analysis/chart stages.

### Guardrails

| Guardrail | Description |
|---|---|
| Binary UUID Protection | Wraps `BINARY(16)` columns with `HEX()` / `BIN_TO_UUID()` — the #1 source of query errors |
| Hallucination Check | Graph validates join paths before SQL generation |
| Syntax Correction | Auto-adds `LIMIT 10`, converts raw SQL text into tool calls |
| Data Verification | Detects binary garbage / empty results, forces a `HEX()` retry |
| Org Scoping | Enforces organization-based row-level security (`WHERE organization.org_id = {org_id}`) |

### Multi-turn memory

LangGraph's `MemorySaver` checkpointing is configured with `thread_id = session_id`. Conversation history persists across messages in the same session. Session IDs are stored in the frontend's localStorage and routed via the API.

---

## SchemaRAG (FAISS)

At startup, every database table is indexed into a FAISS vector store. Each document combines two parts:

**Business context** — Hand-written descriptions: table roles ("contributor_type is the master dictionary of conditions like Anxiety, Diabetes"), medical terminology ("SDOH = Social Determinants of Health"), and query pattern workflows (step-by-step instructions for diagnosis/insurance/organization queries). The `TABLE_TO_CONTEXTS` mapping associates each table with relevant context tags.

**Technical schema** — `CREATE TABLE` DDL with column names and types.

The combined document is embedded via `text-embedding-004` and stored in FAISS. At query time, `search_tables(query)` performs similarity search returning the top-K most semantically relevant tables. Because business context is indexed alongside technical schema, a search for "insurance type" returns `fct_patient_metrics` (which has `insurance_name`) even though "type" does not appear in the column name.

Marts are indexed alongside staging tables, so the RAG returns the right tier for each query.

---

## SchemaGraph (NetworkX)

A NetworkX graph of the database schema — nodes are tables, edges are JOIN relationships with `on` clauses and weights. Edges come from two sources: auto-discovered foreign keys from `information_schema.KEY_COLUMN_USAGE` (weight 1.0), and manually injected edges for logical relationships not enforced by constraints (7 hardcoded paths through bridge tables).

`find_connection_query(table_names)` anchors on `patient`, runs Dijkstra shortest path to every other table, and assembles a `FROM ... JOIN ... ON ...` clause. Used only at Tier 3 — queries the marts cannot answer.

---

## Sandbox

All agent-generated Python code runs inside isolated Docker containers. The `SandboxExecutor` class wraps docker-py:

**Image:** Python 3.12-slim + matplotlib, non-root user.

**Per execution:** A temp directory is created. Code is written to `script.py`. A container runs with `--network none` (no internet), `--memory=256m`, `--cpus=1`, `--pids-limit=50`, 30-second timeout. The DuckDB database is mounted read-only. stdout, stderr, exit code, and any artifacts (chart PNGs) are captured. The container and temp directory are cleaned up.

**Tool integration:** `run_python_code_in_sandbox` (arbitrary Python), `analyze_data` (statistics using Python's `statistics` module), `generate_chart` (matplotlib, 5 chart types, base64 PNG output). All three execute through the sandbox.

**Security properties:** No network access prevents exfiltration. Read-only DB mount prevents data modification. Resource caps prevent exhaustion. Non-root user prevents privilege escalation.

---

## API

FastAPI on port 8000. CORS enabled. Agent initialized once at startup via lifespan context manager.

| Endpoint | Auth | Rate Limit | Purpose |
|---|---|---|---|
| `GET /health` | None | None | Agent status |
| `POST /auth/login` | None | None | Returns JWT access + refresh tokens |
| `POST /auth/refresh` | None | None | Renews access token |
| `POST /chat` | Bearer JWT | 20 req/min/user | Main query endpoint |

### Authentication

Passwords hashed with bcrypt via `passlib`. JWT tokens signed with HS256 via `python-jose`. Access token: 30-minute expiry. Refresh token: 7-day expiry. Tokens carry `sub` (username), `org_id`, and `exp` claims. A FastAPI `Depends(get_current_user)` validates the Bearer header and returns the token payload. `org_id` flows from JWT claims to the agent for row-level security filtering in SQL.

### Rate limiting

An in-memory `RateLimiter` class maintains a dictionary mapping user keys to lists of request timestamps. `is_allowed(key)` purges timestamps older than a 60-second sliding window, checks whether the remaining count exceeds 20, and appends the current timestamp if allowed. Per-key isolation prevents cross-user interference. A `lru_cache`'d factory creates a singleton instance. Applied as a dependency on `/chat`.

---

## Frontend

React 18 + TypeScript + Vite + TailwindCSS v4. State management via Zustand.

**Component tree:** `App → ChatInterface → ChatMessage (×N) + ChatInput`

**ChatInterface:** On mount, health-checks backend, initializes session ID from localStorage. Manages message list, loading state, connection status. Navbar with CaliperLens branding, message area with empty state or scrollable message list, footer with connection badge + input.

**ChatInput:** Textarea with auto-resize, Enter submits, Shift+Enter for newline.

**ChatMessage:** Avatar + styled bubble (blue for user, gray for AI, red for errors). Renders inline chart images from base64 PNG data when present.

**Zustand store:** `messages`, `isLoading`, `connectionStatus`, `sessionId` + actions. `clearMessages` resets state with a new session ID.

**API service:** Native `fetch`. `API_URL` from `VITE_API_URL` env var, defaults to `localhost:8000`.

---

## Analysis & Charts

Two tools execute via the sandbox using pre-defined code templates in `backend/core/analysis.py`:

**`analyze_data`:** Accepts a JSON array of numbers. Template computes mean, median, stddev (Python `statistics` module), min, max, count. Includes trend detection (linear regression on recent values → "increasing"/"decreasing"/"stable"). Returns JSON.

**`generate_chart`:** Accepts `[{label, value}]` arrays + chart type + title. Template generates a matplotlib figure (Agg backend), renders the selected chart type (bar/line/scatter/pie/histogram), saves to an in-memory buffer, base64-encodes the PNG. Returns JSON with `chart_data_base64`.

Both use placeholder-based code templates to avoid f-string escaping issues. Output contracts validated via Pydantic schemas (`AnalysisResult`, `ChartResult`).

---

## Observability

**Structured logging:** `python-json-logger` produces JSON log lines. A `TraceInjector` filter automatically adds `trace_id` (per-query UUID), `session_id`, and `node` (current LangGraph node) to every record. Logs go to daily files and stdout.

**Prometheus:** `prometheus-fastapi-instrumentator` adds a `/metrics` endpoint exposing `http_requests_total` and `http_request_duration_seconds`. Scraped by Prometheus every 15s.

**Grafana:** Pre-built dashboard with p95 latency, request rate, and error rate panels (port 3001).

**LangSmith:** Full graph execution tracing when `LANGCHAIN_TRACING_V2=true`. Node-by-node traces, tool calls with inputs/outputs, retry loops. No code changes needed — LangChain auto-integrates.

---

## Evaluation Harness

25 questions across six categories (simple_select, filtered_select, aggregation, join, multi_step, edge_case) in `eval/questions.json`. Each question specifies expected SQL fragments, result shape (scalar or rows with min/max bounds), and required tables.

**Runner modes:**

- `--check` (CI-safe): Validates question schema. No database needed. Runs on every push.
- `--run` (requires DB + API key): Imports the agent, runs each question via `run_with_trace()`, validates generated SQL structure and result shape. Reports pass-rate and auto-repair-rate (retried-and-passed / retried).

The agent's `run_with_trace()` method captures generated SQL queries, raw SQL results, and retry count during graph streaming — data the standard `run()` method discards.

---

## Infrastructure

**Dockerfile:** Python 3.12-slim, uv-based install, uvicorn on :8000.

**docker-compose.yaml:** Backend + Prometheus + Grafana. Backend mounts `/var/run/docker.sock` for spawning sandbox containers.

**Airflow compose (separate):** `airflow/docker-compose.airflow.yaml` — webserver, scheduler, postgres. Custom image with dbt-duckdb installed. DAG triggers `dbt run` daily.

**CI (GitHub Actions):** Three jobs — backend (ruff + pytest), frontend (tsc + eslint + vitest), eval (`--check`).

**Makefile:** `setup`, `dev`, `test`, `style`, `build`, `clean`, `infra-up`, `infra-down`, `dbt-run`, `eval`, `help`.

---

## Boundaries

- MySQL dump is input-only, never modified.
- dbt transforms are idempotent and re-runnable.
- Frontend is a static Vercel shell — no backend connection in production.
- Sandbox uses Docker-out-of-Docker (`/var/run/docker.sock` mount).
- Rate limiter is in-memory (single-instance).
- License: BUSL-1.1 (converts to Apache 2.0 on 2030-01-01).
