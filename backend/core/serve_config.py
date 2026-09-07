"""Serve config — arch §5.3/§5.5/§5.10/§5.12. One architecture, two sizings."""

SERVE_CONFIG = {
    "fargate": {"min_tasks": 3, "max_tasks": 10, "streams_per_task": 15, "cpu": "1 vCPU"},
    "lambda": {"timeout_s": 30, "memory_mb": 512, "reserved_concurrency": 100, "pilot_reserved": 10},
    "bedrock": {"base_tpm": 150000},
}


def bedrock_tpm(peak: int = 50, calls_per_run: int = 2, tokens_per_call: int = 2500, mean_call_s: float = 2.0) -> int:
    # quota math §5.5: 50*2*2.5k/2 ≈125k + headroom →150k
    raw = peak * calls_per_run * tokens_per_call / mean_call_s
    return int(raw * 1.2)  # headroom
