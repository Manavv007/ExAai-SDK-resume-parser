"""Application settings loaded from environment."""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


def _apply_dotenv_overrides() -> None:
    """Make project .env win over pre-existing shell GOOGLE_API_KEY / GEMINI_API_KEY."""
    # google-genai prefers GOOGLE_API_KEY; drop stale shell value not present in .env.
    os.environ.pop("GOOGLE_API_KEY", None)
    if not _ENV_FILE.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(_ENV_FILE, override=True)
    except ImportError:
        pass


ScreeningMode = Literal["pipeline", "agent"]
LlmProvider = Literal["gemini", "openrouter", "groq", "auto"]
SandboxProvider = Literal["cloud_run", "docker", "e2b", "upstash_box"]
SandboxNetworkMode = Literal["none", "install_only", "always"]
SandboxPreRunMode = Literal["auto", "none", "risk_only", "full"]
ResolvedSandboxPreRunMode = Literal["none", "risk_only", "full"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
    )

    gemini_api_key: str = ""
    gemini_model_id: str = "gemini-2.0-flash"
    gemini_use_vertexai: bool = Field(
        default=False,
        description=(
            "Use Vertex AI (Application Default Credentials) instead of AI Studio GEMINI_API_KEY. "
            "Requires VERTEX_GCP_PROJECT_ID (or GCP_PROJECT_ID) and GCP_REGION; set "
            "GOOGLE_APPLICATION_CREDENTIALS or run gcloud auth application-default login."
        ),
    )
    vertex_gcp_project_id: str = Field(
        default="",
        description=(
            "GCP project for Vertex AI when GEMINI_USE_VERTEXAI=true. "
            "Use when LLM runs on a different project than Cloud Run sandbox (GCP_PROJECT_ID)."
        ),
    )
    vertex_google_application_credentials: str = Field(
        default="",
        description=(
            "Service account JSON for Vertex AI. Defaults to GOOGLE_APPLICATION_CREDENTIALS."
        ),
    )
    sandbox_google_application_credentials: str = Field(
        default="",
        description=(
            "Service account JSON for Cloud Run sandbox + GCS reports on GCP_PROJECT_ID. "
            "Use when Vertex and sandbox use different GCP projects/accounts."
        ),
    )
    llm_provider: LlmProvider = Field(
        default="auto",
        description=(
            "auto: Groq if GROQ_API_KEY set, else OpenRouter if OPEN_ROUTER_API_KEY set, "
            "else Gemini"
        ),
    )
    groq_api_key: str = ""
    groq_model_id: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model for pipeline scoring (LiteLLM groq/ prefix added automatically)",
    )
    groq_agent_model_id: str = Field(
        default="llama-3.3-70b-versatile",
        description=(
            "Groq model for ADK agent tool calling (use 70b; 8b-instant often tool_use_failed)"
        ),
    )
    groq_fallback_model_ids: str = Field(
        default="llama-3.1-8b-instant",
        description="Comma-separated fallback Groq models after 429 on primary",
    )
    groq_max_agent_turns: int = Field(
        default=3,
        description="Max ADK LLM turns when using Groq (free tier RPM/TPM limits)",
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
    github_clone_analysis_enabled: bool | str = Field(
        default="auto",
        description=(
            "Clone selected GitHub repositories and evaluate them in a sandbox. "
            "Supports True, False, or 'auto' for dynamic hybrid evaluation."
        ),
    )
    sandbox_provider: SandboxProvider = Field(
        default="cloud_run",
        description="Sandbox backend for repository execution analysis.",
    )
    sandbox_max_repos: int = Field(
        default=12,
        description="Legacy max selected repositories to clone/evaluate per candidate.",
    )
    sandbox_max_resume_repos: int = Field(
        default=12,
        description=(
            "Max resume-mentioned GitHub repository URLs to analyze/sandbox "
            "(typical resumes list up to ~6)."
        ),
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
    sandbox_deferred_enabled: bool = Field(
        default=False,
        description=(
            "When false (default), sandbox finishes before scoring and POST /screen "
            "returns only completed results. When true, return a provisional score "
            "with status processing and finalize in the background."
        ),
    )
    sandbox_overlap_enabled: bool = Field(
        default=False,
        description=(
            "When true (and deferred mode is off), run sandbox evaluation in parallel "
            "with agent/pipeline scoring; await reports at submit/score time."
        ),
    )
    sandbox_llm_scoring_enabled: bool = Field(
        default=True,
        description=(
            "When true, the agent/scorer judges sandbox reports (vulns, secrets, severe issues) "
            "instead of applying deterministic test/CI penalties after submit."
        ),
    )
    agent_evidence_orchestration_enabled: bool = Field(
        default=False,
        description=(
            "When true with SCREENING_MODE=agent, the agent calls get_github_repo_structures, "
            "fetch_profiles, and run_sandbox_analysis instead of prep-time sandbox."
        ),
    )
    sandbox_focus_max_files: int = Field(
        default=12,
        description="Maximum focused file excerpts returned per sandboxed repository.",
    )
    sandbox_top_files_count: int = Field(
        default=5,
        description="Number of compacted top files per repo in sandbox top_files payload.",
    )
    sandbox_pre_run_mode: SandboxPreRunMode = Field(
        default="auto",
        description=(
            "Prep-time sandbox before the agent when AGENT_EVIDENCE_ORCHESTRATION_ENABLED=true. "
            "auto: none under agent orchestration, full otherwise; "
            "none: agent must call run_sandbox_analysis with focus_paths; "
            "risk_only: vuln/secret pre-pass before agent (no file excerpts); "
            "full: legacy pre-run with heuristic file sampling."
        ),
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

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _strip_llm_provider(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

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

    @field_validator("github_clone_analysis_enabled", mode="before")
    @classmethod
    def _coerce_github_clone_analysis_enabled(cls, value: object) -> object:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                return True
            if lowered in ("false", "0", "no", "off"):
                return False
            if lowered in ("auto", "hybrid"):
                return lowered
        return value

    jd_parse_use_llm: bool = Field(
        default=False,
        description="Use Gemini for JD structuring; false = heuristic only (saves 1 API call)",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "LLM sampling temperature for scoring and structured JSON calls (0 = most stable)."
        ),
    )
    scoring_score_step: int = Field(
        default=5,
        ge=1,
        le=25,
        description=(
            "Quantize requirement match_score and final overall score to this step "
            "(e.g. 5 -> 70, 75, 80)."
        ),
    )
    scoring_rubric_derived: bool = Field(
        default=True,
        description=(
            "When true, overall resume_similarity_score is the weighted rubric mean "
            "from requirement_matches (ignores LLM overall score inflation)."
        ),
    )

    infer_profile_urls: bool = False
    auto_enrich_profiles: bool = Field(
        default=True,
        description=(
            "Pre-fetch resume profile URLs via Exa before agent scoring. "
            "Populates enriched_contents and sources_crawled even when the agent "
            "skips fetch_profiles."
        ),
    )
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
    screening_result_store_path: str = "./data/screening-results"

    max_agent_turns: int = Field(
        default=8,
        description="Max LLM round-trips per ADK agent screening run",
    )
    agent_run_timeout_seconds: int = Field(
        default=120,
        description=(
            "Wall-clock timeout for a single agent screening run. When agent evidence "
            "orchestration is enabled, the effective timeout is raised to cover sandbox wait."
        ),
    )

    agent_version: str = "0.1.0"
    log_level: str = "INFO"
    log_format: str = Field(
        default="text",
        description="Logging format: text (default) or json",
    )
    agent_trace_enabled: bool = Field(
        default=False,
        description=(
            "Enable detailed trace logs for agent lifecycle, tool calls, and enrichment "
            "decision paths."
        ),
    )
    host: str = "0.0.0.0"
    port: int = 8080

    def parsed_api_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


def resolve_vertex_gcp_project(settings: Settings) -> str:
    """Vertex AI project (may differ from sandbox Cloud Run project)."""
    return settings.vertex_gcp_project_id.strip() or settings.gcp_project_id.strip()


def get_settings() -> Settings:
    """Load settings from environment on each call (.env edits apply without cache)."""
    from agent.llm_client import sync_llm_env

    _apply_dotenv_overrides()
    settings = Settings()
    sync_llm_env(settings)
    return settings


def clear_settings_cache() -> None:
    """Backward-compatible no-op (settings are not cached)."""


get_settings.cache_clear = clear_settings_cache  # type: ignore[attr-defined]


def _bootstrap_env() -> None:
    """Sync .env into os.environ as early as possible (before ADK reads credentials)."""
    _apply_dotenv_overrides()
    from agent.llm_client import sync_llm_env

    sync_llm_env(Settings())


_bootstrap_env()
