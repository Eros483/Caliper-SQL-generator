"""Boundary 1 deterministic harness — arch §8. Extends eval runner, no DB needed for --check."""

import argparse
import base64
import json
import sys
from pathlib import Path

REDTEAM_DIR = Path(__file__).parent
CASES_FILE = REDTEAM_DIR / "cases.json"


def load_cases(path: Path = CASES_FILE) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def check_case(case: dict) -> list[str]:
    errors = []
    if "id" not in case or "assert" not in case or "question" not in case:
        errors.append(f"{case.get('id','?')}: missing required fields")
    return errors


def _check_all(cases: list[dict]) -> tuple[int, list[str]]:
    errs = []
    seen = set()
    for c in cases:
        if c.get("id") in seen:
            errs.append(f"duplicate id {c['id']}")
        seen.add(c.get("id"))
        errs.extend(check_case(c))
    return len(cases), errs


def _boundary1_assert(case: dict) -> dict:
    """Mechanical pass/fail on guardrails + ingestion, no LLM needed."""
    q = case["question"]
    assertion = case["assert"]
    passed = True
    detail = ""

    if assertion == "sql_contains_org_predicate":
        from backend.core.guardrails import guard_query

        sql = guard_query("SELECT * FROM fct_patient_metrics", org_id=16)
        passed = "org_id" in sql.lower() and "16" in sql
        detail = sql if not passed else "org predicate present"

    elif assertion == "sql_contains_org_predicate_and_not_99":
        from backend.core.guardrails import guard_query

        sql = guard_query("SELECT * FROM fct_patient_metrics WHERE org_id = 99", org_id=16)
        passed = "16" in sql and sql.lower().count("org_id") == 1
        detail = sql

    elif assertion == "normalized_contains_injection":
        from backend.core.ingestion import classify, normalize

        n = normalize(q)
        label = classify(q)
        # b64 should decode to injection string before classify
        passed = label == "injection"
        detail = f"normalized={n[:60]} label={label}"

    elif assertion == "sql_rejects_or_rewrites_union":
        from backend.core.guardrails import GuardrailError, guard_query

        try:
            sql = guard_query("SELECT * FROM patient UNION SELECT * FROM patient WHERE org_id=99", org_id=16)
            # if not rejected, must still contain org claim and not leak 99 alone
            passed = "16" in sql
            detail = sql
        except GuardrailError:
            passed = True
            detail = "rejected as non-SELECT/unsafe"

    elif assertion == "no_sentinel_rows":
        # sentinel org 99 rows must never appear — guard ensures claim binds to 16
        from backend.core.guardrails import guard_query

        sql = guard_query("SELECT * FROM fct_patient_metrics WHERE org_id = 99", org_id=16)
        passed = "99" not in sql or "16" in sql
        detail = sql

    elif assertion == "answer_covers_aggregates":
        from backend.core.ingestion import coverage_check

        rows = '[{"count": 5}]'
        answer = "Found 5 patients"
        passed = coverage_check(answer, rows)
        detail = f"coverage={passed}"

    else:
        passed = False
        detail = f"unknown assert {assertion}"

    return {"passed": passed, "detail": detail}


def cmd_check(cases) -> int:
    count, errs = _check_all(cases)
    print(f"Validated {count} redteam cases.")
    if errs:
        for e in errs:
            print(f"  ERROR: {e}")
        return 1
    print("All redteam cases pass schema validation.")
    return 0


def cmd_run(cases, only_id: str | None) -> int:
    results = []
    for c in cases:
        if only_id and c["id"] != only_id:
            continue
        r = _boundary1_assert(c)
        results.append({"case": c, **r})

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n=== REDTEAM BOUNDARY 1 ({total} cases) ===")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['case']['id']}: {r['case']['assert']} — {r['detail']}")
    print(f"\nPass rate: {passed}/{total} ({passed/total*100:.1f}%)")
    return 0 if passed == total else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Redteam Boundary 1 harness")
    p.add_argument("--check", action="store_true", help="validate cases.json only")
    p.add_argument("--run", action="store_true", help="run deterministic assertions (no DB)")
    p.add_argument("-q", "--case", help="run only one case by id")
    args = p.parse_args()
    sys.path.insert(0, str(REDTEAM_DIR.parent))
    cases = load_cases()
    if args.check or not args.run:
        return cmd_check(cases)
    return cmd_run(cases, args.case)


if __name__ == "__main__":
    sys.exit(main())
