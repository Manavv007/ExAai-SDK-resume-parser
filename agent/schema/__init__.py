"""Resume screening result schema and models."""

from pathlib import Path

from agent.schema.models import (
    CrawledSource,
    RedFlag,
    RequirementMatch,
    RequirementType,
    ResumeScreeningResult,
    ResumeScreeningStatus,
    ResumeSimilarityScore,
    ScreeningError,
    ScreeningMetadata,
    ScreeningRecommendation,
    SourceRelevance,
)

SCHEMA_VERSION = "1.0"
SCHEMA_ID = "resume-screening-result-v1"
SCHEMA_PATH = Path(__file__).resolve().parent / f"{SCHEMA_ID}.json"

__all__ = [
    "SCHEMA_ID",
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "CrawledSource",
    "RedFlag",
    "RequirementMatch",
    "RequirementType",
    "ResumeScreeningResult",
    "ResumeScreeningStatus",
    "ResumeSimilarityScore",
    "ScreeningError",
    "ScreeningMetadata",
    "ScreeningRecommendation",
    "SourceRelevance",
]
