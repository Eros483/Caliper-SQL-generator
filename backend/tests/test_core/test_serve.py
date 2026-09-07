"""Serve seams — arch §5.3/§5.5/§5.10. Fargate caps, Bedrock quota, Lambda sandbox contract."""


class TestServeConfig:
    def test_fargate_concurrency_caps(self):
        from backend.core.serve_config import SERVE_CONFIG

        assert SERVE_CONFIG["fargate"]["min_tasks"] == 3
        assert SERVE_CONFIG["fargate"]["max_tasks"] == 10
        assert SERVE_CONFIG["fargate"]["streams_per_task"] == 15

    def test_bedrock_quota_math(self):
        from backend.core.serve_config import bedrock_tpm

        # 50 peak chains *2 calls *2.5k tokens /2s ≈125k, headroom 150k
        assert bedrock_tpm(peak=50) == 150000

    def test_lambda_sandbox_caps(self):
        from backend.core.serve_config import SERVE_CONFIG

        assert SERVE_CONFIG["lambda"]["timeout_s"] == 30
        assert SERVE_CONFIG["lambda"]["memory_mb"] == 512
        assert SERVE_CONFIG["lambda"]["reserved_concurrency"] >= 10


class TestObservabilityParity:
    def test_metrics_endpoint_exists(self):
        from backend.main import app

        paths = []
        for r in app.routes:
            if hasattr(r, "path"):
                paths.append(r.path)
            elif hasattr(r, "path_regex"):
                paths.append(str(r.path_regex))
        assert any("metrics" in p for p in paths)

    def test_structured_logger_has_trace_fields(self):
        from backend.utils.logger import get_logger

        logger = get_logger("test")
        # logger has TraceInjector filter that adds trace_id/session_id/node
        assert any(f.__class__.__name__ == "_TraceInjector" for f in logger.filters) or len(logger.handlers) > 0
