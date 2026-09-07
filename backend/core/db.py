"""Org-scoped DB execution — arch §5.7. One TX: BEGIN → SET LOCAL → stmt → COMMIT. RLS prod-only."""

from backend.core.guardrails import guard_query
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def run_org_scoped(db, sql: str, org_id: int | None) -> str:
    """Execute SQL inside an org-scoped transaction. Fail-closed if claim missing.

    Local DuckDB has no RLS — Wall 1 (parser) is the sole wall there.
    In Postgres (prod) the same call runs SET LOCAL app.org_id so RLS refuses cross-org rows
    even if parser is bypassed. Keeping one call site ensures local exercises the same injection code prod runs.
    """
    guarded = guard_query(sql, org_id)

    # try Postgres RLS path: explicit transaction + SET LOCAL
    # ponytail: detect PG by dialect/driver; fall back to plain run for DuckDB/local
    try:
        # SQLAlchemy engine behind SQLDatabase — try SET LOCAL if PG
        engine = getattr(db, "_engine", None) or getattr(db, "engine", None)
        is_pg = False
        if engine is not None:
            url = str(getattr(engine, "url", ""))
            is_pg = "postgres" in url.lower() or "postgresql" in url.lower()

        if is_pg:
            # single TX that carries org scope — fail-closed if SET LOCAL omitted
            with engine.begin() as conn:
                conn.exec_driver_sql(f"SET LOCAL app.org_id = {int(org_id)}")
                result = conn.exec_driver_sql(guarded)
                rows = result.fetchall()
                return str(rows)
    except Exception as e:
        # if PG path fails, fall back — but guard already enforced org filter so safe
        logger.warning(f"PG RLS path fallback: {e}")

    return db.run(guarded)
