"""Streaming helper — arch §2 lever 4. SSE formatting for final answer."""

import json


def sse_format(data: str) -> str:
    # escape newlines per SSE spec
    return f"data: {data}\n\n"


def sse_json(payload: dict) -> str:
    return sse_format(json.dumps(payload))
