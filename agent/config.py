"""Application settings loaded from environment."""

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ScreeningMode = Literal["pipeline", "agent"]
LlmProvider = Literal["gemini", "openrouter", "auto"]
SandboxProvider = Literal["cloud_run", "docker", "e2b", "upstash_box"]
SandboxNetworkMode = Literal["none", "install_only", "always"]


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
        description="OpenRouter model for pipeline scoring",
    )
    openrouter_agent_model_id: str = Field(
        default="openai/gpt-oss-20b:free",
        description="OpenRouter model for ADK agent tool calling",
    )
    openrouter_fallback_model_ids: str = Field(
        default="openai/gpt-oss-20b:free",
        description="Comma-separated fallback OpenRouter models after 429 on primary",
    )
    openrouter_free_max_agent_turns: int = Field(
        default=3,
        description="Max ADK LLM turns when using OpenRouter free models (each turn = 1 API call)",
    )
    exa_api_key: str = ""
    github_token: str = ""
    github_analysis_enabled: bool = True
    github_llm_summary_enabled: bool = False
    max_repos_to_analyze: int = 3
    max_files_per_repo: int = 15
    github_content_token_cap: int = 12000
    github_api_timeout_seconds: int = 10
    github_clone_analysis_enabled: bool = Field(
        default=False,
        description="Clone selected GitHub repositories and evaluate them in a sandbox.",
    )
    sandbox_provider: SandboxProvider = Field(
        default="cloud_run",
        description="Sandbox backend for repository execution analysis.",
    )
    sandbox_max_repos: int = Field(
        default=2,
        description="Legacy max selected repositories to clone/evaluate per candidate.",
    )
    sandbox_max_resume_repos: int = Field(
        default=5,
        description="Max resume-mentioned GitHub repositories to sandbox/evaluate.",
    )
    sandbox_max_profile_repos: int = Field(
        default=2,
        description="Max ranked public profile repositories to sandbox/evaluate as fallback.",
    )
    sandbox_timeout_seconds: int = Field(
        default=300,
        description="Wall-clock timeout for one repository sandbox evaluation.",
    )
    sandbox_wait_seconds: float = Field(
        default=45.0,
        description="Max seconds the screening flow waits for sandbox reports.",
    )
    sandbox_poll_interval_seconds: float = Field(
        default=2.0,
        description="Polling interval for Cloud Run sandbox job operations.",
    )
    sandbox_network_mode: SandboxNetworkMode = Field(
        default="install_only",
        description=(
            "none: no network; install_only: network for dependency install only; "
            "always: allow network."
        ),
    )
    gcp_project_id: str = ""
    gcp_region: str = "asia-south1"
    cloud_run_sandbox_job_name: str = "repo-evaluator"
    sandbox_report_bucket: str = ""
    sandbox_report_prefix: str = "sandbox-reports"
    api_keys: str = Field(default="", description="Comma-separated Bearer tokens")

    screening_mode: ScreeningMode = Field(
        default="pipeline",
        description="agent: ADK Runner tools; pipeline: enrich-all + score",
    )

    @field_validator("screening_mode", mode="before")
    @classmethod
    def _strip_screening_mode(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("sandbox_provider", "sandbox_network_mode", mode="before")
    @classmethod
    def _strip_sandbox_literals(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

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
    url_fetch_concurrency: int = Field(
        default=12,
        description="Max concurrent outbound URL fetches (semaphore) for enrichment",
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


def get_settings() -> Settings:
    """Load settings from environment on each call (.env edits apply without cache)."""
    from agent.llm_client import sync_llm_env

    settings = Settings()
    sync_llm_env(settings)
    return settings


def clear_settings_cache() -> None:
    """Backward-compatible no-op (settings are not cached)."""


get_settings.cache_clear = clear_settings_cache  # type: ignore[attr-defined]
