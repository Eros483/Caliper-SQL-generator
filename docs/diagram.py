"""CaliperLens architecture diagrams via mingrammer/diagrams.

Generates:
  docs/images/aws_architecture.png  — AWS prod (architecture.md §3)
  docs/images/local_architecture.png — Local Docker dev (design.md)

Run:  cd backend && uv run python ../docs/diagram.py
"""

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.compute import ECS, Fargate, Lambda
from diagrams.aws.database import Aurora, RDS
from diagrams.aws.management import AmazonManagedGrafana, AmazonManagedPrometheus, Cloudtrail, Cloudwatch
from diagrams.aws.ml import Bedrock
from diagrams.aws.network import APIGateway, CloudFront, ELB
from diagrams.aws.security import Cognito, KMS, SecretsManager, WAF
from diagrams.aws.storage import S3
from diagrams.onprem.client import Users
from diagrams.onprem.compute import Server
from diagrams.onprem.database import MySQL, Postgresql
from diagrams.programming.framework import FastAPI, React

OUT = Path(__file__).parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

graph_attr = {"fontsize": "16", "bgcolor": "white", "pad": "0.4", "splines": "ortho", "ranksep": "0.9", "nodesep": "0.5"}
node_attr = {"fontsize": "10"}
edge_attr = {"fontsize": "8"}

# ── AWS PROD  ──  TB: Edge → ALB → Fargate, then 3 columns below
with Diagram(
    "CaliperLens — AWS Production  (architecture.md §3)",
    filename=str(OUT / "aws_architecture"),
    show=False,
    outformat="png",
    direction="TB",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
):
    clinician = Users("Clinician\n21 orgs • 200 users • 30-50 peak")

    with Cluster("Edge — CloudFront + WAF + API Gateway  (§5.1)"):
        cf = CloudFront("CloudFront\nTLS + React build (S3)")
        waf = WAF("WAF\nIP rate + managed rules")
        apigw = APIGateway("API Gateway\nper-key throttling")
        cf >> waf >> apigw

    alb = ELB("ALB\n30s hard cap → 503 + trace ID")

    with Cluster("VPC  —  Private  (PHI never leaves)"):
        # top row: serving + LLM
        with Cluster("Serving — Fargate  (§5.3)\n 3-10 tasks × 15 SSE streams • stateless"):
            fargate = Fargate("Fargate API\nFastAPI")
            langgraph = Server("LangGraph\nplanner → generate → check\n→ run_tools → validate → final\nretry×3 • 30s cap")

        with Cluster("LLM — Bedrock  (§5.5)\n VPC endpoint • prompt-cache • ~150k TPM"):
            bedrock = Bedrock("Bedrock\nHaiku (reason) + Titan (embed)\nBAA • in-account")

        with Cluster("Sandbox — Lambda  (§5.10)\n512 MB • 30 s • no egress • zero DB role"):
            sandbox = Lambda("Lambda\ncharts / stats\n≤200 KB rows payload\nmatplotlib layer")

        # middle row: data
        with Cluster("Data — Aurora + S3 + dbt  (§5.6 / §5.7)\n Dual-layer org_id (parser injection + RLS) • RDS Proxy cap 50"):
            aurora_mysql = Aurora("Aurora MySQL\nsource of truth")
            s3 = S3("S3 Parquet Lake\nSSE-KMS • per-org • versioned\n100 GB • 3× headroom")
            dbt = ECS("ECS Cron\n dbt-duckdb\n nightly • Parquet →\n Postgres ATTACH")
            pg_primary = RDS("Aurora Postgres\nPRIMARY\n dbt writes • PITR 35d • RPO ≤5m")
            pg_replica = RDS("Aurora Postgres\nREPLICA\n chat reads • RLS • pgvector HNSW")
            proxy = Server("RDS Proxy")
            vector = Postgresql("pgvector HNSW\n+ NetworkX (7 bridge edges)\nsearch_tables() top-K • tier hint")
            cache = Postgresql("Semantic Cache\n hash(normalized q + org + mart_ver)\n hit → ~1.5 s, zero LLM calls")

        with Cluster("Identity & Secrets  (§5.2 / §5.11 / §5.13)"):
            cognito = Cognito("Cognito\nper-org groups → org_id\nJWT 30m / 7d • JWKS cache")
            secrets = SecretsManager("Secrets Manager\nDB creds (rotated)")
            kms = KMS("KMS  CMKs\n Aurora • S3 • Logs")

        with Cluster("Observability — 3 planes  (§5.12)"):
            cw = Cloudwatch("CloudWatch + X-Ray\nJSON logs + traces • KMS\n trace_id / session / node")
            amp = AmazonManagedPrometheus("AMP\nremote-write /metrics")
            grafana = AmazonManagedGrafana("Managed Grafana\np95 • TTFT • cost • hit-rate")
            trail = Cloudtrail("CloudTrail\n+ S3 access logs")
            audit = S3("Audit Log\nappend-only (INSERT only)\nObject Lock 7 yr • per-org")
            langsmith = Server("LangSmith\nfull traces (BAA)\nnode + tool + retry")

    # ── Edges ──
    clinician >> Edge(label="HTTPS") >> cf >> waf >> apigw >> alb >> fargate
    fargate >> Edge(label="thread_id = session_id\nowner check") >> langgraph

    langgraph >> Edge(label="VPC endpoint") >> bedrock
    langgraph >> Edge(label="SET LOCAL app.org_id\nRLS fail-closed", color="firebrick") >> proxy >> pg_replica
    langgraph >> Edge(label="≤200 KB rows", style="dashed") >> sandbox
    langgraph >> Edge(label="cosine top-K + Dijkstra", style="dotted") >> vector
    langgraph >> Edge(label="hit → 1.5 s", color="darkgreen", style="bold") >> cache

    pg_primary >> Edge(label="repl  •  lag >30s alarm", style="dotted") >> pg_replica
    dbt >> Edge(label="4 marts • fct_patient_metrics") >> pg_primary
    s3 >> Edge(label="Parquet") >> dbt
    aurora_mysql >> Edge(label="nightly extract") >> s3
    proxy >> pg_primary
    proxy >> pg_replica

    fargate >> Edge(label="verify JWKS\n0 Cognito calls/req") >> cognito
    fargate >> Edge(style="dotted") >> secrets
    kms >> Edge(style="dotted") >> pg_primary
    kms >> Edge(style="dotted") >> s3

    fargate >> Edge(label="/metrics (same name)\nasync batch", color="darkgreen") >> amp >> grafana
    fargate >> Edge(label="JSON logs") >> cw
    fargate >> Edge(label="INSERT-only", style="bold") >> audit
    fargate >> Edge(style="dashed") >> langsmith
    trail >> Edge(style="dotted") >> s3

# ── LOCAL DEV ──
with Diagram(
    "CaliperLens — Local Docker Dev  (design.md)",
    filename=str(OUT / "local_architecture"),
    show=False,
    outformat="png",
    direction="LR",
    graph_attr={"fontsize": "15", "bgcolor": "white", "pad": "0.4", "splines": "ortho", "ranksep": "0.8", "nodesep": "0.5"},
    node_attr={"fontsize": "10"},
    edge_attr={"fontsize": "8"},
):
    dev = Users("Developer\nlocalhost (HIPAA)")

    with Cluster("Docker Host  —  docker-compose  (Vercel = static shell only)"):
        fe = React("React 18 + Vite\nTailwind v4 • Zustand\n:5173  •  prod = HIPAA notice")
        be = FastAPI("FastAPI :8000\nlifespan → SQLAgentGenerator\nCORS + /health")
        agent = Server("LangGraph Agent\n6 nodes • retry×3\nGemini 3.5 Flash • text-embedding-004")

        with Cluster("Data — MySQL → dbt → DuckDB"):
            mysql = MySQL("MySQL\nfhs_coredb_local\n(dump, input-only)")
            dbt_local = Server("dbt-duckdb\nstaging → inter → marts\n10 + 4 + 4 • idempotent")
            duckdb = Postgresql("DuckDB\ncaliperlens.duckdb\nfct_patient_metrics")
            airflow = Server("Airflow\nDAG daily")

        with Cluster("Retrieval  (local = FAISS)"):
            faiss = Server("FAISS\ntext-embedding-004\nbusiness context + DDL\n→ top-K + tier hint")
            graph = Server("NetworkX\nSchemaGraph • 7 bridges\nDijkstra • Tier 3 only")

        with Cluster("Sandbox — Docker"):
            docker_sandbox = Server("Docker\n--network none • 256m/1cpu/30s\nro /data • matplotlib")

        with Cluster("Observability (local)"):
            prom = Server("Prometheus :9090\nscrape /metrics 15s")
            graf = Server("Grafana :3001\np95 • req rate • err rate")
            logs = Server("JSON logs + LangSmith flag")

    dev >> Edge(label=":5173") >> fe >> Edge(label=":8000 /api/v1") >> be >> agent
    agent >> Edge(label="RAG top-K") >> faiss
    agent >> Edge(style="dotted") >> graph
    agent >> Edge(style="dashed") >> docker_sandbox
    mysql >> Edge(label="LOAD mysql; ATTACH") >> dbt_local >> duckdb
    airflow >> Edge(style="dotted") >> dbt_local
    be >> Edge(label="/metrics") >> prom >> graf
    be >> logs

print(f"✓ Diagrams written to {OUT}/")
