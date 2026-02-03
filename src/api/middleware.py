"""
FastAPI middleware stack.

  1. CORSMiddleware   – allows cross-origin requests from the Streamlit UI
                        (and any future frontend).
  2. RequestIDMiddleware – stamps every request with a unique X-Request-ID
                           header for tracing.
  3. LoggingMiddleware   – logs method, path, status, and latency for every
                           request.
  4. ResponseCacheMiddleware – Redis-backed GET cache (opt-in per route via
                               a custom header or query param).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import redis
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from src.config import settings

logger = logging.getLogger("automotive_gpt.middleware")

# ─── CORS origins ────────────────────────────────────────────
# In production, lock this down to your actual domain(s).
ALLOWED_ORIGINS = [
    "http://localhost:8501",   # Streamlit dev
    "http://localhost:3000",   # Any future React frontend
    "*",                       # Remove in production!
]


# ─────────────────────────────────────────────────────────────
# Request-ID middleware
# ─────────────────────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every incoming request."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ─────────────────────────────────────────────────────────────
# Logging middleware
# ─────────────────────────────────────────────────────────────

class LoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status code, and latency."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        request_id = getattr(request.state, "request_id", "?")
        logger.info(
            "[%s] %s %s → %d (%.1f ms)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


# ─────────────────────────────────────────────────────────────
# Redis response cache
# ─────────────────────────────────────────────────────────────

_DEFAULT_CACHE_TTL = 300  # 5 minutes


class ResponseCacheMiddleware(BaseHTTPMiddleware):
    """
    Simple GET-request cache backed by Redis.

    Cached routes must be idempotent (GET only).  The cache key is the
    full URL path + query string.  POST requests and non-200 responses
    are never cached.

    To skip the cache for a specific request, include the header:
        X-Cache: no-cache
    """

    def __init__(self, app, ttl: int = _DEFAULT_CACHE_TTL):
        super().__init__(app)
        self.ttl = ttl
        try:
            self.redis = redis.from_url(settings.redis_url, decode_responses=True)
            self.redis.ping()
            self._enabled = True
        except Exception as e:
            logger.warning("Redis cache unavailable: %s. Caching disabled.", e)
            self._enabled = False

    async def dispatch(self, request: Request, call_next):
        if not self._enabled or request.method != "GET":
            return await call_next(request)

        # Check for cache-skip header
        if request.headers.get("x-cache", "").lower() == "no-cache":
            return await call_next(request)

        cache_key = f"response_cache:{request.url.path}?{request.url.query}"

        # Try cache hit
        cached = self.redis.get(cache_key)
        if cached:
            logger.debug("Cache HIT: %s", cache_key)
            return JSONResponse(content=_safe_json_loads(cached))

        # Cache miss — call the route
        response = await call_next(request)

        # Only cache 200 responses
        if response.status_code == 200:
            # Read body (streaming response)
            body_chunks = []
            async for chunk in response.body_iterator:
                body_chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
            body = b"".join(body_chunks)

            # Store in Redis
            try:
                self.redis.setex(cache_key, self.ttl, body.decode())
                logger.debug("Cache SET: %s (TTL=%ds)", cache_key, self.ttl)
            except Exception as e:
                logger.warning("Cache write failed: %s", e)

            # Return a new response with the same body
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        return response


def _safe_json_loads(s: str) -> Any:
    import json
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {"raw": s}
