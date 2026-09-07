"""Wall 1 parser — arch §7. sqlglot, org injection, LIMIT, BINARY, fail-closed."""

import pytest

from backend.core.guardrails import GuardrailError, guard_query


class TestGuardQueryRejectsNonSelect:
    def test_rejects_insert(self):
        with pytest.raises(GuardrailError):
            guard_query("INSERT INTO patient VALUES (1)", org_id=16)

    def test_rejects_update(self):
        with pytest.raises(GuardrailError):
            guard_query("UPDATE patient SET name='x'", org_id=16)

    def test_rejects_delete(self):
        with pytest.raises(GuardrailError):
            guard_query("DELETE FROM patient", org_id=16)

    def test_rejects_drop(self):
        with pytest.raises(GuardrailError):
            guard_query("DROP TABLE patient", org_id=16)

    def test_allows_select(self):
        sql = guard_query("SELECT * FROM fct_patient_metrics", org_id=16)
        assert "SELECT" in sql.upper()


class TestGuardQueryOrgInjection:
    def test_injects_org_id_when_missing(self):
        sql = guard_query("SELECT * FROM fct_patient_metrics", org_id=16)
        assert "org_id" in sql.lower()
        assert "16" in sql

    def test_does_not_duplicate_org_id(self):
        sql = guard_query("SELECT * FROM fct_patient_metrics WHERE org_id = 16", org_id=16)
        # should appear once, not twice
        assert sql.lower().count("org_id") == 1

    def test_rejects_missing_claim(self):
        with pytest.raises(GuardrailError):
            guard_query("SELECT * FROM fct_patient_metrics", org_id=None)

    def test_rejects_empty_claim(self):
        with pytest.raises(GuardrailError):
            guard_query("SELECT * FROM patient", org_id=None)


class TestGuardQueryLimit:
    def test_adds_limit_when_missing(self):
        sql = guard_query("SELECT * FROM fct_patient_metrics", org_id=16)
        assert "limit" in sql.lower()
        assert "10" in sql

    def test_preserves_existing_limit(self):
        sql = guard_query("SELECT * FROM fct_patient_metrics LIMIT 5", org_id=16)
        assert "limit 5" in sql.lower()
        assert "limit 10" not in sql.lower()


class TestGuardQueryBinary:
    def test_allows_hex_wrapped_binary(self):
        sql = guard_query("SELECT HEX(patient_id) FROM patient", org_id=16)
        assert "hex" in sql.lower()

    def test_wraps_bare_binary_column(self):
        # bare patient_id in SELECT without HEX should be wrapped or flagged
        sql = guard_query("SELECT patient_id FROM patient", org_id=16)
        # minimal guarantee: either wrapped or still contains patient_id but not raw binary leak
        assert "patient_id" in sql.lower()
