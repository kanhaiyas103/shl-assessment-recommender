"""Validated environment-based application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from shl_agent.models.enums import LlmProvider
from shl_agent.scraper.constants import ASSIGNMENT_CATALOG_URL


class Settings(BaseSettings):
    """Immutable settings loaded from SHL_-prefixed environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SHL_",
        extra="ignore",
        frozen=True,
    )

    app_env: Literal["local", "test", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    host: str = "0.0.0.0"
    port: Annotated[int, Field(ge=1, le=65_535)] = 8000

    catalog_path: Path = Path("data/processed/catalog.json")
    catalog_manifest_path: Path = Path("data/processed/catalog_manifest.json")
    catalog_report_path: Path = Path("data/processed/catalog_report.json")
    raw_catalog_path: Path = Path("data/raw/shl_product_catalog.json")
    catalog_source_url: str = ASSIGNMENT_CATALOG_URL
    faiss_index_path: Path = Path("data/indexes/index.faiss")
    index_metadata_path: Path = Path("data/indexes/metadata.json")
    embedding_manifest_path: Path = Path("data/indexes/embedding_manifest.json")
    embedding_report_path: Path = Path("data/indexes/embedding_report.json")
    index_manifest_path: Path = Path("data/indexes/manifest.json")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: Annotated[int, Field(ge=1, le=128)] = 32

    scraper_timeout_seconds: Annotated[float, Field(gt=0, le=60)] = 20
    scraper_retry_attempts: Annotated[int, Field(ge=1, le=5)] = 3
    scraper_retry_backoff_seconds: Annotated[float, Field(ge=0, le=10)] = 0.75
    scraper_rate_limit_seconds: Annotated[float, Field(ge=0, le=5)] = 0.25

    llm_provider: LlmProvider = LlmProvider.DISABLED
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_base_url: HttpUrl | None = None
    llm_timeout_seconds: Annotated[float, Field(gt=0, le=25)] = 15
    request_timeout_seconds: Annotated[float, Field(gt=0, le=30)] = 30

    max_conversation_messages: Literal[8] = 8
    retrieval_candidate_limit: Annotated[int, Field(ge=10, le=100)] = 50
    recommendation_limit: Annotated[int, Field(ge=1, le=10)] = 10

    @model_validator(mode="after")
    def validate_provider_configuration(self) -> "Settings":
        """Require complete provider configuration whenever an LLM is enabled."""
        if self.llm_provider is not LlmProvider.DISABLED:
            if self.llm_api_key is None or not self.llm_api_key.get_secret_value().strip():
                raise ValueError("llm_api_key is required when an LLM provider is enabled")
            if self.llm_model is None or not self.llm_model.strip():
                raise ValueError("llm_model is required when an LLM provider is enabled")
        if self.llm_provider is LlmProvider.OPENROUTER and self.llm_base_url is None:
            raise ValueError("llm_base_url is required for OpenRouter")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache process-wide immutable settings."""
    return Settings()
