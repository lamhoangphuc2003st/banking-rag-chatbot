from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from apps.api.app.core.logging import get_logger

logger = get_logger(__name__)


class CacheBackend(Protocol):
    async def get(self, key: str) -> Any | None: ...

    async def set(self, key: str, value: Any) -> None: ...

    async def clear(self) -> None: ...

    async def warmup(self) -> None: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class _CacheEntry:
    value: Any
    expires_at: float


class AsyncTTLCache:
    def __init__(self, *, max_entries: int, ttl_seconds: float) -> None:
        self.max_entries = max(1, max_entries)
        self.ttl_seconds = max(0.0, ttl_seconds)
        self._entries: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        if self.ttl_seconds <= 0:
            return None

        now = time.monotonic()
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._entries.pop(key, None)
                return None
            return entry.value

    async def set(self, key: str, value: Any) -> None:
        if self.ttl_seconds <= 0:
            return

        now = time.monotonic()
        async with self._lock:
            self._entries[key] = _CacheEntry(
                value=value,
                expires_at=now + self.ttl_seconds,
            )
            self._prune(now)

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        await self.clear()

    def _prune(self, now: float) -> None:
        expired_keys = [
            key
            for key, entry in self._entries.items()
            if entry.expires_at <= now
        ]
        for key in expired_keys:
            self._entries.pop(key, None)

        overflow = len(self._entries) - self.max_entries
        if overflow <= 0:
            return

        for key in list(self._entries)[:overflow]:
            self._entries.pop(key, None)


class RedisTTLCache:
    def __init__(
        self,
        *,
        redis_url: str,
        namespace: str,
        ttl_seconds: float,
        encode: Callable[[Any], str],
        decode: Callable[[str], Any],
        socket_connect_timeout_seconds: float = 5.0,
        socket_timeout_seconds: float = 5.0,
        socket_keepalive: bool = True,
        health_check_interval_seconds: float = 30.0,
        retry_on_timeout: bool = True,
    ) -> None:
        self.redis_url = redis_url
        self.namespace = namespace.strip(":")
        self.ttl_seconds = max(0.0, ttl_seconds)
        self.encode = encode
        self.decode = decode
        self.socket_connect_timeout_seconds = max(0.1, socket_connect_timeout_seconds)
        self.socket_timeout_seconds = max(0.1, socket_timeout_seconds)
        self.socket_keepalive = socket_keepalive
        self.health_check_interval_seconds = max(0.0, health_check_interval_seconds)
        self.retry_on_timeout = retry_on_timeout
        self._redis: Any | None = None

    async def get(self, key: str) -> Any | None:
        if self.ttl_seconds <= 0:
            return None

        try:
            raw_value = await self._client().get(self._redis_key(key))
        except Exception as exc:  # pragma: no cover - provider boundary
            logger.warning("redis_cache_get_failed", namespace=self.namespace, error=str(exc))
            return None

        if raw_value is None:
            return None
        try:
            return self.decode(str(raw_value))
        except Exception as exc:
            logger.warning("redis_cache_decode_failed", namespace=self.namespace, error=str(exc))
            return None

    async def set(self, key: str, value: Any) -> None:
        if self.ttl_seconds <= 0:
            return

        try:
            payload = self.encode(value)
            await self._client().setex(
                self._redis_key(key),
                max(1, int(self.ttl_seconds)),
                payload,
            )
        except Exception as exc:  # pragma: no cover - provider boundary
            logger.warning("redis_cache_set_failed", namespace=self.namespace, error=str(exc))

    async def clear(self) -> None:
        try:
            redis = self._client()
            keys = [key async for key in redis.scan_iter(f"{self.namespace}:*")]
            if keys:
                await redis.delete(*keys)
        except Exception as exc:  # pragma: no cover - provider boundary
            logger.warning("redis_cache_clear_failed", namespace=self.namespace, error=str(exc))

    async def warmup(self) -> None:
        """Eagerly open the connection so the first user request doesn't pay the
        cold connect cost. Best-effort: a failure here is logged, never raised."""
        if self.ttl_seconds <= 0:
            return
        try:
            await self._client().ping()
        except Exception as exc:  # pragma: no cover - provider boundary
            logger.warning("redis_cache_warmup_failed", namespace=self.namespace, error=str(exc))

    async def close(self) -> None:
        client = self._redis
        self._redis = None
        if client is not None:
            await client.aclose()

    def _client(self) -> Any:
        if self._redis is None:
            from redis.asyncio import Redis

            self._redis = Redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=self.socket_connect_timeout_seconds,
                socket_timeout=self.socket_timeout_seconds,
                socket_keepalive=self.socket_keepalive,
                health_check_interval=int(self.health_check_interval_seconds),
                retry_on_timeout=self.retry_on_timeout,
            )
        return self._redis

    def _redis_key(self, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return f"{self.namespace}:{digest}"
