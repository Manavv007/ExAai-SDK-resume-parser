"""Application settings loaded from environment."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ScreeningMode = Literal["pipeline", "agent"]
LlmProvider = Literal["gemini", "openrouter", "auto"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    gemini_model_id: str = "gemini-2.0-flash"
    llm_provider: LlmProvider = Field(
        default="auto",
        description="auto: OpenRouter if OPEN_ROUTER_API_KEY set, else Gemini",
    )
    open_router_api_key: str = ""
    openrouter_model_id: str = Field(
        default="openrouter/free",
        description="OpenRouter model id (openrouter/ prefix added automatically)",
    )
    openrouter_fallback_model_ids: str = Field(
        default="openai/gpt-oss-20b:free",
        description="Comma-separated fallback OpenRouter models after 429 on primary",
    )
    llm_max_retries: int = Field(
        default=3,
        description="Retries per model on OpenRouter 429 rate limits",
    )
    llm_retry_backoff_seconds: float = Field(
        default=2.0,
        description="Base backoff seconds between OpenRouter rate-limit retries",
    )
    exa_api_key: str = ""
    api_keys: str = Field(default="", description="Comma-separated Bearer tokens")

    screening_mode: ScreeningMode = Field(
        default="agent",
        description="agent: ADK Runner with tools; pipeline: enrich-all + score fallback",
    )

    jd_parse_use_llm: bool = Field(
        default=False,
        description="Use Gemini for JD structuring; false = heuristic only (saves 1 API call)",
    )

    infer_profile_urls: bool = False
    profile_scoring_mode: str = Field(
        default="strict",
        description="strict: limited profiles omit crawl body; balanced: include with warning",
    )

    max_urls_per_resume: int = Field(
        default=10,
        description="Max profile URLs per resume and max unique agent fetches per session",
    )
    url_fetch_timeout_seconds: int = 5
    content_token_cap: int = 8000
    cache_ttl_seconds: int = 86400
    url_cache_path: str = "./data/url_cache.db"

    max_agent_turns: int = Field(
        default=8,
        description="Max LLM round-trips per ADK agent screening run",
    )
    agent_run_timeout_seconds: int = Field(
        default=120,
        description="Wall-clock timeout for a single agent screening run",
    )

    agent_version: str = "0.1.0"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8080

    def parsed_api_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    from agent.llm_client import sync_llm_env

    settings = Settings()
    sync_llm_env(settings)
    return settings
