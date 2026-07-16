from __future__ import annotations

import hashlib
import time
from collections import defaultdict, deque
from typing import Protocol

from apps.api.app.core.logging import get_logger

logger = get_logger(__name__)


class RateLimiter(Protocol):
    async def allow(self, key: str) -> bool: ...

    async def warmup(self) -> None: ...

    async def close(self) -> None: ...


class InMemoryRateLimiter:
    """Local rate limiter.

    Production deployments should replace this with a Redis-backed limiter so limits
    are shared across replicas.
    """

    def __init__(self, limit_per_minute: int) -> None:
        self.limit_per_minute = limit_per_minute
        self._events: dict[str, deque[float]] = defaultdict(deque)

    async def allow(self, key: str) -> bool:
        now = time.monotonic()
        window_start = now - 60
        events = self._events[key]

        while events and events[0] < window_start:
            events.popleft()

        if len(events) >= self.limit_per_minute:
            return False

        events.append(now)
        return True

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None


class RedisRateLimiter:
    """Fixed-window Redis limiter shared across API replicas."""

    def __init__(
        self,
        redis_url: str,
        limit_per_minute: int,
        *,
        socket_connect_timeout_seconds: float = 5.0,
        socket_timeout_seconds: float = 5.0,
        socket_keepalive: bool = True,
        health_check_interval_seconds: float = 30.0,
        retry_on_timeout: bool = True,
    ) -> None:
        from redis.asyncio import Redis

        self.limit_per_minute = limit_per_minute
        self._redis = Redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=max(0.1, socket_connect_timeout_seconds),
            socket_timeout=max(0.1, socket_timeout_seconds),
            socket_keepalive=socket_keepalive,
            health_check_interval=int(max(0.0, health_check_interval_seconds)),
            retry_on_timeout=retry_on_timeout,
        )

    async def allow(self, key: str) -> bool:
        now = int(time.time())
        window = now // 60
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        redis_key = f"rate-limit:{key_hash}:{window}"

        count = await self._redis.incr(redis_key)
        if count == 1:
            await self._redis.expire(redis_key, 90)
        return int(count) <= self.limit_per_minute

    async def warmup(self) -> None:
        """Open the connection at startup so the first request doesn't pay the
        cold connect cost. Best-effort: a failure here is logged, never raised."""
        try:
            await self._redis.ping()
        except Exception as exc:  # pragma: no cover - provider/network boundary
            logger.warning("rate_limiter_warmup_failed", error=str(exc))

    async def close(self) -> None:
        await self._redis.aclose()


def create_rate_limiter(
    *,
    backend: str,
    limit_per_minute: int,
    redis_url: str,
    redis_socket_connect_timeout_seconds: float = 5.0,
    redis_socket_timeout_seconds: float = 5.0,
    redis_socket_keepalive: bool = True,
    redis_health_check_interval_seconds: float = 30.0,
    redis_retry_on_timeout: bool = True,
) -> RateLimiter:
    normalized_backend = backend.strip().casefold()
    if normalized_backend == "redis":
        if not redis_url.strip():
            raise ValueError("REDIS_URL is required when API_RATE_LIMIT_BACKEND=redis.")
        return RedisRateLimiter(
            redis_url,
            limit_per_minute,
            socket_connect_timeout_seconds=redis_socket_connect_timeout_seconds,
            socket_timeout_seconds=redis_socket_timeout_seconds,
            socket_keepalive=redis_socket_keepalive,
            health_check_interval_seconds=redis_health_check_interval_seconds,
            retry_on_timeout=redis_retry_on_timeout,
        )
    if normalized_backend not in {"", "memory", "inmemory", "local"}:
        logger.warning("unknown_rate_limit_backend", backend=backend, fallback="memory")
    return InMemoryRateLimiter(limit_per_minute)
