"""Provider interface for repository sandbox evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agent.sandbox.models import RepoExecutionReport, SandboxStep


@dataclass(frozen=True)
class SandboxCommand:
    """A command to run inside a cloned repository."""

    step: SandboxStep
    command: str


class SandboxProvider(Protocol):
    """Backend capable of evaluating a repository in an isolated environment."""

    async def evaluate_repo(
        self,
        *,
        repo_url: str,
        repo_name: str,
        commands: list[SandboxCommand],
    ) -> RepoExecutionReport:
        """Evaluate ``repo_url`` and return bounded evidence."""
