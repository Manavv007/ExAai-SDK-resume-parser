"""Application settings loaded from environment."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    gemini_model_id: str = "gemini-2.0-flash"
    exa_api_key: str = ""
    api_keys: str = Field(default="", description="Comma-separated Bearer tokens")

    infer_profile_urls: bool = False
    profile_scoring_mode: str = Field(
        default="strict",
        description="strict: limited profiles omit crawl body; balanced: include with warning",
    )

    max_urls_per_resume: int = 10
    url_fetch_timeout_seconds: int = 5
    content_token_cap: int = 8000
    cache_ttl_seconds: int = 86400
    url_cache_path: str = "./data/url_cache.db"

    agent_version: str = "0.1.0"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8080

    def parsed_api_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
