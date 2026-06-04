"""Request ID, Bearer auth, and request timing."""

import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """
    Legacy placeholder: /screen auth is enforced in api.auth.require_api_key
    so Swagger UI can use the visible api_key form field.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        return await call_next(request)


class TimingMiddleware(BaseHTTPMiddleware):
    """Record wall-clock ms on request.state for /screen responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        request.state.elapsed_ms = int((time.perf_counter() - started) * 1000)
        if request.url.path == "/screen":
            response.headers["X-Processing-Time-Ms"] = str(request.state.elapsed_ms)
        return response
