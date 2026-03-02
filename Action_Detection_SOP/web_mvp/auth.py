from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


@dataclass(frozen=True)
class BasicAuthConfig:
    username: str = "admin"
    password: Optional[str] = None
    cookie_name: str = "sop_review_session"
    cookie_max_age_s: int = 12 * 60 * 60  # 12 hours
    allow_unauth_prefixes: tuple[str, ...] = ("/ui/",)
    allow_unauth_paths: tuple[str, ...] = ("/", "/api/health", "/api/auth/login")


def _unauthorized() -> Response:
    return Response(
        content="Unauthorized",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="sop-review"'},
    )


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode((text + pad).encode("ascii"))


def issue_session_token(*, cfg: BasicAuthConfig, username: str) -> str:
    if not cfg.password:
        raise ValueError("Cannot issue session token without password")
    now = int(time.time())
    payload = {"u": username, "exp": now + int(cfg.cookie_max_age_s)}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(cfg.password.encode("utf-8"), raw, hashlib.sha256).digest()
    return f"{_b64url_encode(raw)}.{_b64url_encode(sig)}"


def validate_session_token(*, cfg: BasicAuthConfig, token: str) -> bool:
    if not cfg.password:
        return True
    if not token or "." not in token:
        return False
    left, right = token.split(".", 1)
    try:
        raw = _b64url_decode(left)
        sig = _b64url_decode(right)
    except Exception:
        return False
    expected = hmac.new(cfg.password.encode("utf-8"), raw, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, sig):
        return False
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("u") != cfg.username:
        return False
    exp = payload.get("exp")
    if not isinstance(exp, int):
        return False
    if int(time.time()) > exp:
        return False
    return True


def is_unauth_allowed(*, cfg: BasicAuthConfig, path: str) -> bool:
    if path in cfg.allow_unauth_paths:
        return True
    return any(path.startswith(prefix) for prefix in cfg.allow_unauth_prefixes)


class BasicAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Callable, *, cfg: BasicAuthConfig) -> None:
        super().__init__(app)
        self._cfg = cfg

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self._cfg.password:
            return await call_next(request)

        path = request.url.path or "/"
        if is_unauth_allowed(cfg=self._cfg, path=path):
            return await call_next(request)

        token = request.cookies.get(self._cfg.cookie_name, "")
        if token and validate_session_token(cfg=self._cfg, token=token):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.lower().startswith("basic "):
            return _unauthorized()

        token = auth.split(" ", 1)[1].strip()
        try:
            decoded = base64.b64decode(token).decode("utf-8")
        except Exception:
            return _unauthorized()

        if ":" not in decoded:
            return _unauthorized()

        username, password = decoded.split(":", 1)
        if username != self._cfg.username or password != self._cfg.password:
            return _unauthorized()

        return await call_next(request)
