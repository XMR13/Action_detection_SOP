from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


@dataclass(frozen=True)
class BasicAuthConfig:
    username: str = "admin"
    password: Optional[str] = None


def _unauthorized() -> Response:
    return Response(
        content="Unauthorized",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="sop-review"'},
    )


class BasicAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Callable, *, cfg: BasicAuthConfig) -> None:
        super().__init__(app)
        self._cfg = cfg

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self._cfg.password:
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

