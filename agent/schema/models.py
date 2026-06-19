"""Pydantic models for resume-screening-result-v1."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ResumeScreeningStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ScreeningRecommendation(StrEnum):
    ADVANCE = "advance"
    HOLD = "hold"
    REJECT = "reject"


class RequirementType(StrEnum):
    TECHNICAL_SKILL = "technical_skill"
    SOFT_SKILL = "soft_skill"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    RESPONSIBILITY = "responsibility"


class RedFlagSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SourceRelevance(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResumeSimilarityScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=100)
    reasoning: str = Field(min_length=1, max_length=500)


class IntegrityIndication(StrEnum):
    GOOD = "good"
    BAD = "bad"
    NOT_ENOUGH_EVIDENCE = "not_enough_evidence"


class CandidateIntegrity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall: IntegrityIndication
    github_account_timeline: IntegrityIndication
    linkedin_contact_links: IntegrityIndication
    github_profile_readme_links: IntegrityIndication
    reasoning: str = Field(min_length=1, max_length=500)


class IntegritySignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal_id: str = Field(min_length=1)
    indication: IntegrityIndication
    evidence: str = Field(min_length=1, max_length=500)
    source_urls: list[str] = Field(default_factory=list)


class RequirementMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement: str = Field(min_length=1)
    requirement_type: RequirementType
    match_score: int = Field(ge=0, le=100)
    evidence: str = Field(min_length=1)
    source_quote: str | None = None


class RedFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flag: str = Field(min_length=1)
    severity: RedFlagSeverity
    evidence: str = Field(min_length=1)


class CrawledSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    relevance: SourceRelevance
    title: str | None = None


class TopFileMatchSignal(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class TopFileEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str = Field(min_length=1)
    repo_url: str = Field(min_length=1)
    path: str = Field(min_length=1)
    importance_rank: int = Field(ge=1)
    classification: str | None = None
    language: str | None = None
    compaction_tier: str | None = None
    total_lines: int | None = Field(default=None, ge=0)
    sent_lines: int | None = Field(default=None, ge=0)
    content_status: str | None = None
    jd_criteria: list[str] = Field(default_factory=list)
    match_signal: TopFileMatchSignal
    assessment: str = Field(min_length=1, max_length=500)
    evidence_snippet: str = Field(min_length=1, max_length=200)


class LlmCallTraceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int = Field(ge=1)
    source: str = Field(min_length=1)
    model: str = Field(min_length=1)
    ts_ms: int = Field(ge=0)


class ScreeningMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    model_version: str = Field(min_length=1)
    processed_at: datetime
    resume_text_chars: int = Field(ge=0)
    agent_version: str = Field(min_length=1)
    processing_time_ms: int | None = Field(default=None, ge=0)
    job_desc_version: str | None = None
    llm_calls: int | None = Field(default=None, ge=0)
    llm_call_trace: list[LlmCallTraceEntry] | None = None
    agent_submit_fallback: bool | None = None
    screening_mode: Literal["pipeline", "agent"] | None = None


class ScreeningError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    source_url: str | None = None


class RepoEvaluationScore(BaseModel):
    model_config = ConfigDict(extra="allow")

    url: str | None = None
    repo: str | None = None
    classification: str | None = None
    ownership_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    activity_score: int | None = Field(default=None, ge=0, le=100)
    documentation_score: int | None = Field(default=None, ge=0, le=100)
    collaborators_score: int | None = Field(default=None, ge=0, le=100)
    repo_raw_score: int | None = Field(default=None, ge=0, le=100)
    repo_final_score: int | None = Field(default=None, ge=0, le=100)
    code_quality_score: int | None = Field(default=None, ge=0, le=100)
    code_quality_bonus: int | None = Field(default=None, ge=0)
    is_fork: bool | None = None


class EvaluationBreakdown(BaseModel):
    model_config = ConfigDict(extra="allow")

    jd_fit_score: int = Field(ge=0, le=100)
    repo_portfolio_score: int | None = Field(default=None, ge=0, le=100)
    code_quality_score: int | None = Field(default=None, ge=0, le=100)
    sandbox_penalty: int = Field(default=0, ge=0, le=100)
    risk_ceiling: int | None = Field(default=None, ge=0, le=100)
    ownership_multiplier_avg: float | None = Field(default=None, ge=0.0, le=1.0)
    composite_score: int = Field(ge=0, le=100)
    final_score: int | None = Field(default=None, ge=0, le=100)
    final_score_source: Literal["llm_or_rubric", "evaluation_composite"] | None = None
    blend_weights: dict[str, float] | None = None
    repos: list[RepoEvaluationScore | dict[str, Any]] = Field(default_factory=list)


class ResumeScreeningResult(BaseModel):
    """Platform output contract."""

    model_config = ConfigDict(extra="forbid")

    application_id: UUID
    job_id: UUID
    resume_screening_status: ResumeScreeningStatus
    metadata: ScreeningMetadata
    errors: list[ScreeningError] = Field(default_factory=list)

    resume_similarity_score: ResumeSimilarityScore | None = None
    candidate_integrity: CandidateIntegrity | None = None
    integrity_signals: list[IntegritySignal] = Field(default_factory=list)
    requirement_matches: list[RequirementMatch] = Field(default_factory=list)
    recommendation: ScreeningRecommendation | None = None
    recommendation_reasoning: str | None = None
    sources_crawled: list[CrawledSource] = Field(default_factory=list)
    temp_sandbox_reports: list[dict[str, Any]] | None = None
    top_file_evaluation: list[TopFileEvaluation] = Field(default_factory=list)
    evaluation_breakdown: EvaluationBreakdown | None = None

    @field_validator("recommendation_reasoning")
    @classmethod
    def strip_reasoning(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def validate_status_rules(self) -> "ResumeScreeningResult":
        if self.resume_screening_status == ResumeScreeningStatus.COMPLETED:
            if self.resume_similarity_score is None:
                raise ValueError("resume_similarity_score is required when status is completed")
            if self.recommendation is None:
                raise ValueError("recommendation is required when status is completed")
            if not self.recommendation_reasoning:
                raise ValueError("recommendation_reasoning is required when status is completed")
        elif self.resume_screening_status == ResumeScreeningStatus.FAILED:
            if not self.errors:
                raise ValueError("errors must be non-empty when status is failed")
        return self

    def to_json_dict(self) -> dict:
        """Serialize for API responses (JSON-compatible)."""
        return self.model_dump(mode="json", exclude_none=False)
