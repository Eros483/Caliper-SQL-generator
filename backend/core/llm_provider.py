"""LLM seam — arch §10. Local Gemini default, Bedrock behind LLM_BACKEND flag."""

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class _BedrockStub:
    """Minimal stub with same surface as ChatGoogleGenerativeAI for bedrock backend.

    ponytail: no real Bedrock call — same invoke/bind_tools contract so local tests pass
    and AWS wiring replaces this class without touching call sites. Add real
    ChatBedrock when Bedrock creds exist.
    """

    def invoke(self, messages, **kwargs):
        raise NotImplementedError("Bedrock backend not configured in local env")

    def bind_tools(self, tools, **kwargs):
        # return self so generate_query_node can chain .invoke
        return self


def _get_local_llm():
    from langchain.chat_models import init_chat_model

    if not settings.gemini_api_key:
        # keep previous behavior: agent.py raises; provider also raises if no key
        logger.warning("GEMINI_API_KEY empty — LLM calls will fail")
    return init_chat_model("google_genai:gemini-3.5-flash", temperature=0)


def get_llm():
    """Factory — single seam for all LLM calls. Respects LLM_BACKEND flag."""
    backend = (settings.llm_backend or "local").lower()
    if backend == "bedrock":
        return _BedrockStub()
    return _get_local_llm()
