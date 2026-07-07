import json
from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "local"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    api_cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )
    api_rate_limit_per_minute: int = 60
    api_rate_limit_backend: str = "redis"

    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0.1
    openai_api_key: str | None = None
    litellm_api_key: str | None = None
    cohere_api_key: str | None = None

    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-large"
    reranker_provider: str = "local"
    reranker_model: str = "rerank-v4.0-fast"
    reranker_max_documents: int = 64
    reranker_request_timeout_seconds: float = 4.0

    rag_cache_enabled: bool = True
    rag_cache_backend: str = "redis"
    rag_cache_ttl_seconds: float = 300.0
    rag_cache_max_entries: int = 512
    qdrant_request_timeout_seconds: float = 6.0

    database_url: str = "postgresql+asyncpg://bankbot:bankbot@localhost:5432/bankbot"
    redis_url: str = ""
    redis_socket_connect_timeout_seconds: float = 2.0
    redis_socket_timeout_seconds: float = 5.0
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "vietcombank_public_docs"
    qdrant_recreate_collection: bool = False

    rag_data_root: str | None = None

    vietcombank_base_url: str = "https://www.vietcombank.com.vn"
    vietcombank_exchange_rate_xml_url: str = (
        "https://portal.vietcombank.com.vn/Usercontrols/TVPortal.TyGia/pXML.aspx"
    )
    crawler_user_agent: str = "BankChatbotResearchCrawler/1.0"
    crawler_request_delay_seconds: float = 1.0

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                parsed = json.loads(stripped)
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, value: str) -> str:
        database_url = str(value).strip()
        if database_url.startswith("postgres://"):
            return database_url.replace("postgres://", "postgresql+asyncpg://", 1)
        if database_url.startswith("postgresql://"):
            return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return database_url

    @property
    def is_local(self) -> bool:
        return self.app_env.lower() in {"local", "dev", "development"}


def sync_database_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
