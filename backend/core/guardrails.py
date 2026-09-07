"""Wall 1 parser — arch §7. Minimal sqlglot guard: non-SELECT rejected, org injection, LIMIT, fail-closed."""

import re

import sqlglot
import sqlglot.expressions as exp

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# tables that carry patient/org scope — anything reading them must be filtered
_PATIENT_TABLES = {
    "patient",
    "map_patient_metrics",
    "lob",
    "organization",
    "stg_patient",
    "stg_map_patient_metrics",
    "stg_lob",
    "stg_organization",
    "fct_patient_metrics",
    "dim_patients",
    "fct_interventions",
    "dim_conditions",
    "contributor_individual",
    "contributor_type",
    "patient_score",
}


class GuardrailError(ValueError):
    pass


def _is_patient_touching(sql: str, parsed: exp.Expression | None) -> bool:
    lowered = sql.lower()
    if parsed is not None:
        try:
            tables = {t.name.lower() for t in parsed.find_all(exp.Table)}
            if tables & _PATIENT_TABLES:
                return True
        except Exception:
            pass
    # fallback string search
    return any(tbl in lowered for tbl in _PATIENT_TABLES)


def guard_query(sql: str, org_id: int | None) -> str:
    """Validate and rewrite a proposed query. Fail-closed on any violation."""
    if not sql or not sql.strip():
        raise GuardrailError("Empty query")

    stripped = sql.strip().rstrip(";")

    # missing claim — every patient-touching query needs it
    # check before parse so even malformed SQL with missing claim fails
    parsed: exp.Expression | None = None
    try:
        parsed = sqlglot.parse_one(stripped, read="duckdb")
    except Exception:
        parsed = None

    # non-SELECT rejected — sqlglot check plus keyword fallback
    if parsed is not None:
        if not isinstance(parsed, exp.Select):
            raise GuardrailError("Only SELECT is allowed")
    else:
        if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
            raise GuardrailError("Only SELECT is allowed")

    is_patient = _is_patient_touching(stripped, parsed)

    if is_patient and org_id is None:
        raise GuardrailError("Missing org claim — query rejected")

    # org injection — only if patient-touching and not already filtered
    if is_patient and org_id is not None:
        # already has org_id predicate => don't duplicate
        if "org_id" not in stripped.lower():
            if re.search(r"\bWHERE\b", stripped, re.IGNORECASE):
                stripped = re.sub(
                    r"\bWHERE\b",
                    f"WHERE org_id = {int(org_id)} AND",
                    stripped,
                    count=1,
                    flags=re.IGNORECASE,
                )
            else:
                # inject before LIMIT/ORDER/GROUP if present, else append
                # simplest: append WHERE before any trailing clauses
                m = re.search(r"\b(GROUP BY|ORDER BY|LIMIT|OFFSET)\b", stripped, re.IGNORECASE)
                if m:
                    idx = m.start()
                    stripped = stripped[:idx].rstrip() + f" WHERE org_id = {int(org_id)} " + stripped[idx:]
                else:
                    stripped = stripped + f" WHERE org_id = {int(org_id)}"

    # BINARY handling — wrap bare patient_id selects with HEX if not already wrapped
    # ponytail: minimal — only rewrite SELECT patient_id without HEX/BIN_TO_UUID
    # keep original if already wrapped
    if (
        re.search(r"SELECT\s+patient_id\b", stripped, re.IGNORECASE)
        and "hex(" not in stripped.lower()
        and "bin_to_uuid" not in stripped.lower()
    ):
        stripped = re.sub(
            r"SELECT\s+patient_id\b",
            "SELECT HEX(patient_id) AS patient_id",
            stripped,
            count=1,
            flags=re.IGNORECASE,
        )
    # also handle SELECT *, patient_id variants — leave for now (covered by HEX check on result path)

    # LIMIT enforcement
    if not re.search(r"\bLIMIT\b", stripped, re.IGNORECASE):
        stripped = stripped + " LIMIT 10"

    return stripped
