from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    api_cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    api_rate_limit_per_minute: int = 60

    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0.1
    openai_api_key: str | None = None
    litellm_api_key: str | None = None

    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-large"
    reranker_provider: str = "local"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    database_url: str = "postgresql+asyncpg://bankbot:bankbot@localhost:5432/bankbot"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "vietcombank_public_docs"

    vietcombank_base_url: str = "https://www.vietcombank.com.vn"
    crawler_user_agent: str = "BankChatbotResearchCrawler/1.0"
    crawler_request_delay_seconds: float = 1.0

    otel_exporter_otlp_endpoint: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def is_local(self) -> bool:
        return self.app_env.lower() in {"local", "dev", "development"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
