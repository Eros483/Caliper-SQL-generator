import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from backend.utils.config import settings

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security_scheme = HTTPBearer()


class TokenPayload(BaseModel):
    sub: str
    org_id: int | None = None
    exp: int


def create_access_token(user_id: str, org_id: int | None = None) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = TokenPayload(sub=user_id, org_id=org_id, exp=int(expire.timestamp()))
    return jwt.encode(payload.model_dump(), settings.secret_key, algorithm=ALGORITHM)


def create_refresh_token(user_id: str, org_id: int | None = None) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = TokenPayload(sub=user_id, org_id=org_id, exp=int(expire.timestamp()))
    return jwt.encode(payload.model_dump(), settings.secret_key, algorithm=ALGORITHM)


def verify_token(token: str) -> TokenPayload:
    # seam: AUTH_BACKEND selects verification path; single call site per arch §10
    # ponytail: cognito path is HS256 fallback until JWKS is wired — add RS256/JWKS when Cognito exists
    backend = (settings.auth_backend or "local").lower()
    if backend == "cognito":
        # TODO: verify via Cognito JWKS (RS256); fallback to HS256 for local parity
        pass
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        return TokenPayload(**payload)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
) -> TokenPayload:
    return verify_token(credentials.credentials)


# Simple in-memory rate limiter using a dict
class RateLimiter:
    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._store: dict[str, list[float]] = {}

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        window_start = now - self.window
        timestamps = [t for t in self._store.get(key, []) if t > window_start]
        self._store[key] = timestamps
        if len(timestamps) >= self.max_requests:
            return False
        timestamps.append(now)
        return True


@lru_cache
def get_rate_limiter() -> RateLimiter:
    return RateLimiter()


async def rate_limit_middleware(
    request: Request,
    limiter: RateLimiter = Depends(get_rate_limiter),
    user: TokenPayload | None = None,
) -> None:
    # Use JWT sub if authenticated, otherwise fall back to IP
    key = user.sub if user else request.client.host if request.client else "unknown"
    if not limiter.is_allowed(key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
