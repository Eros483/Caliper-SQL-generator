from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.core.auth import (
    TokenPayload,
    create_access_token,
    create_refresh_token,
    get_current_user,
    get_rate_limiter,
    verify_password,
    verify_token,
)
from backend.schemas.chat import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    LoginRequest,
    RefreshRequest,
    TokenResponse,
)

router = APIRouter()


def get_agent():
    from backend.main import _agent

    return _agent


@router.get("/health", response_model=HealthResponse)
async def health_check():
    agent = get_agent()
    agent_status = "loaded" if agent else "not loaded"
    overall = "healthy" if agent else "unhealthy"
    return HealthResponse(status=overall, agent=agent_status)


@router.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    from backend.utils.config import settings

    if not settings.demo_user or not settings.demo_password_hash:
        raise HTTPException(status_code=503, detail="Auth not configured")

    if request.username != settings.demo_user or not verify_password(request.password, settings.demo_password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(request.username, org_id=16)
    refresh = create_refresh_token(request.username, org_id=16)
    return TokenResponse(access_token=token, refresh_token=refresh)


@router.post("/auth/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest):
    payload = verify_token(request.refresh_token)
    token = create_access_token(payload.sub, org_id=payload.org_id)
    refresh = create_refresh_token(payload.sub, org_id=payload.org_id)
    return TokenResponse(access_token=token, refresh_token=refresh)


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    user: TokenPayload = Depends(get_current_user),
    limiter=Depends(get_rate_limiter),
):
    agent = get_agent()
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    key = user.sub
    if not limiter.is_allowed(key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        trace = agent.run_with_trace(request.query, session_id=request.session_id, org_id=user.org_id)
        return ChatResponse(response=trace["response"], success=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    user: TokenPayload = Depends(get_current_user),
    limiter=Depends(get_rate_limiter),
):
    from backend.core.streaming import sse_format

    agent = get_agent()
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    if not limiter.is_allowed(user.sub):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    def _gen():
        trace = agent.run_with_trace(request.query, session_id=request.session_id, org_id=user.org_id)
        for token in trace["response"].split(" "):
            yield sse_format(token + " ")
        yield sse_format("[DONE]")

    return StreamingResponse(_gen(), media_type="text/event-stream")
