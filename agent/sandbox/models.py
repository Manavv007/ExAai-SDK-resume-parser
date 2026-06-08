"""Typed results produced by repository sandbox evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SandboxStep = Literal["clone", "install", "build", "test", "runtime", "inspect"]
FindingSeverity = Literal["info", "warn", "high"]
FindingCategory = Literal[
    "structure",
    "quality",
    "tests",
    "dependencies",
    "execution",
    "risk",
]
FailureType = Literal[
    "clone_failure",
    "timeout",
    "resource_killed",
    "missing_system_dependency",
    "dependency_build_failure",
    "install_failure",
    "build_failure",
    "test_failure",
    "runtime_failure",
    "inspection_failure",
]

MAX_OUTPUT_CHARS = 4000
MAX_SUMMARY_CHARS = 1000


def truncate_output(value: str, *, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """Keep command logs bounded before storing or passing them to an LLM."""
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    suffix = f"\n...[truncated {len(text) - max_chars} chars]"
    return text[: max(0, max_chars - len(suffix))] + suffix


class CommandResult(BaseModel):
    """One sandbox command outcome."""

    step: SandboxStep
    command: str
    ok: bool
    exit_code: int | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    failure_type: FailureType | None = None

    @field_validator("stdout", "stderr", mode="before")
    @classmethod
    def _truncate_logs(cls, value: object) -> str:
        return truncate_output(str(value or ""))

    @field_validator("command", mode="before")
    @classmethod
    def _strip_command(cls, value: object) -> str:
        return str(value or "").strip()

    def compact(self) -> dict[str, Any]:
        """Return a compact representation for prompts and audit logs."""
        data: dict[str, Any] = {
            "step": self.step,
            "command": self.command,
            "ok": self.ok,
        }
        if self.exit_code is not None:
            data["exit_code"] = self.exit_code
        if self.duration_ms is not None:
            data["duration_ms"] = self.duration_ms
        if self.error:
            data["error"] = self.error
        if self.failure_type:
            data["failure_type"] = self.failure_type
        output = "\n".join(part for part in (self.stdout, self.stderr) if part).strip()
        if output:
            data["output_preview"] = truncate_output(output, max_chars=1200)
        return data


class RepoExecutionReport(BaseModel):
    """Bounded evidence from cloning and evaluating one public repository."""

    repo: str
    url: str
    provider: str = "cloud_run"
    clone_ok: bool = False
    detected_stack: list[str] = Field(default_factory=list)
    repo_profile: dict[str, Any] = Field(default_factory=dict)
    commands: list[CommandResult] = Field(default_factory=list)
    quality_signals: dict[str, bool | int | str | list[str]] = Field(default_factory=dict)
    risk_flags: list[str] = Field(default_factory=list)
    findings: list[dict[str, str]] = Field(default_factory=list)
    summary: str = ""
    overall_assessment: str = ""
    confidence: Literal["low", "medium", "high"] = "low"
    timed_out: bool = False
    skipped_reason: str | None = None

    @field_validator("summary", mode="before")
    @classmethod
    def _truncate_summary(cls, value: object) -> str:
        return truncate_output(str(value or ""), max_chars=MAX_SUMMARY_CHARS)

    @property
    def install_ok(self) -> bool | None:
        return self._step_ok("install")

    @property
    def build_ok(self) -> bool | None:
        return self._step_ok("build")

    @property
    def tests_ok(self) -> bool | None:
        return self._step_ok("test")

    def _step_ok(self, step: SandboxStep) -> bool | None:
        matches = [command.ok for command in self.commands if command.step == step]
        if not matches:
            return None
        return all(matches)

    def compact(self) -> dict[str, Any]:
        """Return prompt-safe sandbox evidence."""
        data: dict[str, Any] = {
            "repo": self.repo,
            "url": self.url,
            "provider": self.provider,
            "clone_ok": self.clone_ok,
            "detected_stack": self.detected_stack,
            "repo_profile": self.repo_profile,
            "commands": [command.compact() for command in self.commands],
            "quality_signals": self.quality_signals,
            "risk_flags": self.risk_flags,
            "findings": self.findings,
            "summary": self.summary,
            "overall_assessment": self.overall_assessment,
            "confidence": self.confidence,
            "timed_out": self.timed_out,
        }
        if self.skipped_reason:
            data["skipped_reason"] = self.skipped_reason
        return data
