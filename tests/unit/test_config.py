import pytest

from apps.api.app.core.config import Settings
from apps.api.app.core.rate_limit import create_rate_limiter
from apps.api.app.rag.retrieval.hybrid import HybridRetriever


def test_settings_accept_comma_separated_cors_origins() -> None:
    settings = Settings(
        _env_file=None,
        api_cors_origins="http://localhost:3000,http://localhost:3001",
    )

    assert settings.api_cors_origins == ["http://localhost:3000", "http://localhost:3001"]


def test_settings_accept_json_cors_origins() -> None:
    settings = Settings(
        _env_file=None,
        api_cors_origins='["http://localhost:3000", "http://localhost:3001"]',
    )

    assert settings.api_cors_origins == ["http://localhost:3000", "http://localhost:3001"]


def test_settings_normalize_render_postgres_url_to_async_driver() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgres://user:pass@host:5432/db",
    )

    assert settings.database_url == "postgresql+asyncpg://user:pass@host:5432/db"


def test_settings_normalize_plain_postgresql_url_to_async_driver() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql://user:pass@host:5432/db",
    )

    assert settings.database_url == "postgresql+asyncpg://user:pass@host:5432/db"


def test_redis_rate_limiter_requires_redis_url() -> None:
    with pytest.raises(ValueError, match="REDIS_URL is required"):
        create_rate_limiter(
            backend="redis",
            limit_per_minute=60,
            redis_url="",
        )


def test_redis_rag_cache_requires_redis_url() -> None:
    with pytest.raises(ValueError, match="REDIS_URL is required"):
        HybridRetriever(Settings(_env_file=None, rag_cache_backend="redis", redis_url=""))


def test_redis_timeouts_are_configurable() -> None:
    settings = Settings(
        _env_file=None,
        redis_socket_connect_timeout_seconds=3,
        redis_socket_timeout_seconds=8,
    )

    assert settings.redis_socket_connect_timeout_seconds == 3
    assert settings.redis_socket_timeout_seconds == 8


def test_redis_connection_hardening_defaults() -> None:
    settings = Settings(_env_file=None)

    # 2s was too tight for a cold TLS connect after an idle spin-down.
    assert settings.redis_socket_connect_timeout_seconds == 5.0
    assert settings.redis_socket_keepalive is True
    assert settings.redis_health_check_interval_seconds == 30.0
    assert settings.redis_retry_on_timeout is True
