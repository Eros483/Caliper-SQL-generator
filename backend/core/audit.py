"""Append-only audit — arch §5.12 compliance plane. Local file, prod S3 Object Lock."""

import json
import os
from datetime import datetime, timezone

from backend.utils.logger import get_logger

logger = get_logger(__name__)


def audit_log(question: str, sql: str, rows: str, org_id: int | None, trace_id: str = "-") -> None:
    path = os.environ.get("AUDIT_PATH", "logs/audit.jsonl")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "org_id": org_id,
        "question": question,
        "sql": sql,
        "rows_chars": len(rows) if rows else 0,
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(f"audit {trace_id} org={org_id} q={question[:60]}")
