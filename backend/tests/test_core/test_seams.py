"""Parity seams — arch §10. Local default, AWS behind flag. Same sig, both backends pass local."""

from backend.utils.config import Settings


class TestSettingsParityFlags:
    def test_llm_backend_flag_exists(self):
        s = Settings()
        assert hasattr(s, "llm_backend")
        assert s.llm_backend == "local"

    def test_auth_backend_flag_exists(self):
        s = Settings()
        assert hasattr(s, "auth_backend")
        assert s.auth_backend == "local"

    def test_sandbox_backend_flag_exists(self):
        s = Settings()
        assert hasattr(s, "sandbox_backend")
        assert s.sandbox_backend == "local"

    def test_vector_backend_flag_exists(self):
        s = Settings()
        assert hasattr(s, "vector_backend")
        assert s.vector_backend == "local"

    def test_env_flag_exists(self):
        s = Settings()
        assert hasattr(s, "env")
        assert s.env in ("development", "local", "production", "test")

    def test_flags_read_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "bedrock")
        monkeypatch.setenv("AUTH_BACKEND", "cognito")
        monkeypatch.setenv("SANDBOX_BACKEND", "lambda")
        monkeypatch.setenv("VECTOR_BACKEND", "pgvector")
        s = Settings()
        assert s.llm_backend == "bedrock"
        assert s.auth_backend == "cognito"
        assert s.sandbox_backend == "lambda"
        assert s.vector_backend == "pgvector"


class TestLLMProviderSeam:
    def test_get_llm_provider_local_uses_gemini(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("LLM_BACKEND", "local")
        from backend.core.llm_provider import get_llm

        llm = get_llm()
        # local backend returns object with invoke/bind_tools (ChatGoogleGenerativeAI compat)
        assert hasattr(llm, "invoke")
        assert hasattr(llm, "bind_tools")

    def test_get_llm_provider_bedrock_stub_has_same_interface(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "bedrock")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        from backend.core.llm_provider import get_llm

        llm = get_llm()
        assert hasattr(llm, "invoke")
        assert hasattr(llm, "bind_tools")


class TestSandboxSeam:
    def test_get_sandbox_executor_local(self, monkeypatch):
        monkeypatch.setenv("SANDBOX_BACKEND", "local")
        from backend.core.sandbox import get_sandbox_executor

        ex = get_sandbox_executor()
        assert hasattr(ex, "execute")
        assert hasattr(ex, "health_check")
        # same contract: execute(code, timeout) -> dict with stdout/stderr/exit_code/artifacts
        assert callable(ex.execute)

    def test_get_sandbox_executor_lambda_has_same_contract(self, monkeypatch):
        monkeypatch.setenv("SANDBOX_BACKEND", "lambda")
        from backend.core.sandbox import get_sandbox_executor

        ex = get_sandbox_executor()
        assert hasattr(ex, "execute")
        assert hasattr(ex, "health_check")

    def test_sandbox_executor_interface_preserved(self):
        from backend.core.sandbox import SandboxExecutor

        ex = SandboxExecutor.__new__(SandboxExecutor)
        assert hasattr(ex, "execute")
        assert hasattr(ex, "health_check")


class TestAuthSeam:
    def test_verify_token_local_still_works(self, monkeypatch):
        monkeypatch.setenv("AUTH_BACKEND", "local")
        monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
        from backend.core.auth import create_access_token, verify_token

        token = create_access_token("user1", org_id=42)
        payload = verify_token(token)
        assert payload.sub == "user1"
        assert payload.org_id == 42

    def test_verify_token_single_call_site(self):
        # arch §10: verify_token stays the single call site regardless of backend
        import backend.core.auth as auth_mod

        assert hasattr(auth_mod, "verify_token")
        assert callable(auth_mod.verify_token)
        assert hasattr(auth_mod, "get_current_user")


class TestVectorSeam:
    def test_search_tables_signature_preserved(self):
        import inspect

        from backend.src.rag_manager import SchemaRAG

        sig = inspect.signature(SchemaRAG.search_tables)
        params = list(sig.parameters)
        assert "query" in params
        assert "k" in params
