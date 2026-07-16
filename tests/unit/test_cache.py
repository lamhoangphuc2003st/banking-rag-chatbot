from __future__ import annotations

from typing import Any

import pytest

from apps.api.app.core.config import Settings
from apps.api.app.core.rate_limit import RedisRateLimiter
from apps.api.app.rag.cache import AsyncTTLCache, RedisTTLCache
from apps.api.app.rag.retrieval.hybrid import HybridRetriever


class FakeRedis:
    """Stand-in for ``redis.asyncio.Redis`` that records connection kwargs and
    PINGs so tests can assert the resilience options and warm-up behaviour
    without a live server."""

    instances: list[FakeRedis] = []

    def __init__(self, url: str, **kwargs: Any) -> None:
        self.url = url
        self.kwargs = kwargs
        self.ping_calls = 0
        self.ping_error: Exception | None = None
        self.closed = False
        FakeRedis.instances.append(self)

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> FakeRedis:
        return cls(url, **kwargs)

    async def ping(self) -> bool:
        self.ping_calls += 1
        if self.ping_error is not None:
            raise self.ping_error
        return True

    async def get(self, key: str) -> Any:
        return None

    async def setex(self, *args: Any) -> None:
        return None

    async def incr(self, key: str) -> int:
        return 1

    async def expire(self, *args: Any) -> None:
        return None

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> type[FakeRedis]:
    FakeRedis.instances = []
    monkeypatch.setattr("redis.asyncio.Redis", FakeRedis)
    return FakeRedis


def _make_cache(**overrides: Any) -> RedisTTLCache:
    params: dict[str, Any] = dict(
        redis_url="redis://localhost:6379/0",
        namespace="bank-chatbot:rag:retrieval",
        ttl_seconds=60.0,
        encode=str,
        decode=str,
    )
    params.update(overrides)
    return RedisTTLCache(**params)


def test_redis_cache_client_uses_resilient_connection_options(
    fake_redis: type[FakeRedis],
) -> None:
    cache = _make_cache(
        socket_connect_timeout_seconds=5.0,
        socket_timeout_seconds=5.0,
        socket_keepalive=True,
        health_check_interval_seconds=30.0,
        retry_on_timeout=True,
    )

    cache._client()

    assert len(fake_redis.instances) == 1
    kwargs = fake_redis.instances[0].kwargs
    assert kwargs["socket_connect_timeout"] == 5.0
    assert kwargs["socket_timeout"] == 5.0
    assert kwargs["socket_keepalive"] is True
    # health_check_interval must be an int — redis-py compares it against a clock.
    assert kwargs["health_check_interval"] == 30
    assert isinstance(kwargs["health_check_interval"], int)
    assert kwargs["retry_on_timeout"] is True


async def test_redis_cache_warmup_opens_connection(fake_redis: type[FakeRedis]) -> None:
    cache = _make_cache()

    await cache.warmup()

    assert len(fake_redis.instances) == 1
    assert fake_redis.instances[0].ping_calls == 1


async def test_redis_cache_warmup_swallows_connection_errors(
    fake_redis: type[FakeRedis],
) -> None:
    cache = _make_cache()
    client = cache._client()
    client.ping_error = TimeoutError("Timeout connecting to server")

    # A cold/unreachable Redis at boot must not crash startup.
    await cache.warmup()

    assert client.ping_calls == 1


async def test_disabled_cache_warmup_does_not_open_connection(
    fake_redis: type[FakeRedis],
) -> None:
    cache = _make_cache(ttl_seconds=0.0)

    await cache.warmup()

    assert fake_redis.instances == []


async def test_async_ttl_cache_warmup_is_noop() -> None:
    cache = AsyncTTLCache(max_entries=8, ttl_seconds=60.0)

    await cache.warmup()  # must not raise

    assert await cache.get("missing") is None


async def test_hybrid_retriever_warmup_opens_all_cache_connections(
    fake_redis: type[FakeRedis],
) -> None:
    retriever = HybridRetriever(
        Settings(
            _env_file=None,
            rag_cache_backend="redis",
            redis_url="redis://localhost:6379/0",
            rag_cache_enabled=True,
            rag_cache_ttl_seconds=60.0,
        )
    )

    await retriever.warmup()

    # retrieval + embedding + scroll — each its own connection, each pinged once.
    assert len(fake_redis.instances) == 3
    assert all(instance.ping_calls == 1 for instance in fake_redis.instances)

    await retriever.close()


async def test_redis_rate_limiter_uses_resilient_connection_options(
    fake_redis: type[FakeRedis],
) -> None:
    limiter = RedisRateLimiter("redis://localhost:6379/0", 60)

    kwargs = fake_redis.instances[0].kwargs
    assert kwargs["socket_connect_timeout"] == 5.0
    assert kwargs["socket_keepalive"] is True
    assert kwargs["health_check_interval"] == 30
    assert kwargs["retry_on_timeout"] is True

    await limiter.warmup()

    assert fake_redis.instances[0].ping_calls == 1
