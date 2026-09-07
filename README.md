<div align="center">

# CaliperLens
</center>

<center>
<p>
    <img src="https://img.shields.io/badge/license-BUSL--1.1-blue" alt="License"/>
    <img src="https://img.shields.io/badge/version-2.0.0-green" alt="Version"/>
    <img src="https://img.shields.io/badge/python-3.12+-blue" alt="Python"/>
    <img src="https://img.shields.io/badge/typescript-5.4+-3178c6" alt="TypeScript"/>
    <img src="https://img.shields.io/badge/react-18.3-61dafb" alt="React"/>
    <img src="https://img.shields.io/badge/llm-gemini_3.5_flash-4285f4" alt="Gemini"/>
    <img src="https://img.shields.io/badge/CI-passing-brightgreen" alt="CI"/>
</p>
</center>

An agentic natural-language-to-SQL engine for querying complex healthcare datasets autonomously. Uses a multi-agent sandboxed workflow to reason through database schemas and generate complex queries.

<div align="left">

## Getting Started

```bash
git clone https://github.com/Eros483/CaliperLens.git
cd CaliperLens
cp .env.example .env
```

Set your `.env` with the required keys (see `.env.example`).

```bash
make setup    # Install all dependencies
make dev      # Start frontend + backend
make test     # Run all tests
make style    # Format + lint
```

For the complete architecture — agent graph, tools exposed to the LLM, guardrails, data pipeline, and infra — see [docs/design.md](docs/design.md).

## Data Pipeline

MySQL holds the raw dump; dbt builds the DuckDB marts the agent queries.

```bash
make infra-up   # Start MySQL + backend + Prometheus + Grafana
make db-load    # Load db-dump.sql into MySQL (one-time, ~25 min for the 1.2G dump)
make dbt-run    # Transform MySQL → DuckDB marts (data/caliperlens.duckdb)
```

## System Flow

The diagram shows the two pipelines that keep CaliperLens running: the **offline data pipeline** (MySQL → DuckDB, scheduled by Airflow) and the **online query pipeline** (user question → validated answer).

```mermaid
flowchart LR
    subgraph DATA["Data engineering pipeline (offline)"]
        direction LR
        MYSQL["MySQL dump"] --> DBT["dbt transforms"] --> DUCK["DuckDB"]
    end

    USER["User"] --> FE["React frontend"] --> AGENT["LangGraph agent"]
    AGENT -->|queries| DUCK
    AGENT --> RAG["FAISS RAG"]
    AGENT --> GRAPH["SchemaGraph"]
    AGENT --> SANDBOX["Docker sandbox"]
    AGENT --> API["FastAPI"] --> FE
```

## Directory Overview

```
├── backend/               # FastAPI + LangGraph agent
│   ├── api/v1/            # Versioned route handlers (thin)
│   ├── core/              # Business logic (auth, sandbox, analysis, planner, tiers)
│   ├── src/               # Agent internals
│   │   ├── agent.py           # Main LangGraph Agent definition
│   │   ├── custom_tools.py    # Tools exposed to the LLM
│   │   ├── graph_manager.py   # NetworkX logic for join path discovery
│   │   ├── prompt_module.py   # System prompts for agent states
│   │   └── rag_manager.py     # FAISS vector store for schema search
│   ├── schemas/           # Pydantic request/response models
│   ├── utils/             # Config, logger, exceptions
│   ├── tests/             # test_api/ + test_core/ (mirror module layout)
│   └── main.py            # FastAPI entry point
│
├── frontend/              # React + Vite + TS chat UI
│   ├── src/
│   │   ├── components/    # ChatInterface, ChatMessage, ChatInput
│   │   ├── store/         # Zustand chat state
│   │   ├── services/      # API call functions
│   │   └── __tests__/     # Vitest component tests
│   ├── public/
│   ├── index.html
│   └── (vite.config.ts, tsconfig.json, package.json)
│
├── dbt/                   # dbt project: staging → intermediate → marts
│   └── models/
│       ├── staging/       # 1:1 MySQL mirrors (views)
│       ├── intermediate/  # pre-computed joins (tables)
│       └── marts/         # analytics-ready models (tables)
│
├── airflow/               # Airflow DAGs + docker-compose for dbt scheduling
├── sandbox/               # Docker image for isolated code execution
├── eval/                  # NL-to-SQL eval harness (questions.json + runner.py)
├── grafana/               # Grafana dashboard provisioning
├── docs/
│   ├── problem.md         # Original problem statement
│   ├── design.md          # Architecture & design document
│   └── features.json      # Feature tracker
│
├── Makefile               # Single entry point for setup/dev/test/style/build
├── docker-compose.yaml    # Backend + Prometheus + Grafana
├── prometheus.yml
└── Dockerfile
```

## License

BUSL-1.1 — free for personal, educational, and portfolio use. Production or commercial deployment requires a separate license. See [LICENSE](LICENSE).
