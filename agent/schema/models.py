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


class ResumeScreeningResult(BaseModel):
    """Platform output contract."""

    model_config = ConfigDict(extra="forbid")

    application_id: UUID
    job_id: UUID
    resume_screening_status: ResumeScreeningStatus
    metadata: ScreeningMetadata
    errors: list[ScreeningError] = Field(default_factory=list)

    resume_similarity_score: ResumeSimilarityScore | None = None
    requirement_matches: list[RequirementMatch] = Field(default_factory=list)
    recommendation: ScreeningRecommendation | None = None
    recommendation_reasoning: str | None = None
    red_flags: list[RedFlag] = Field(default_factory=list)
    sources_crawled: list[CrawledSource] = Field(default_factory=list)
    temp_sandbox_reports: list[dict[str, Any]] | None = None

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
