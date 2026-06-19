"""GitHub repository analysis tool for the screening agent."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import pydantic

from agent.config import get_settings
from agent.logging_config import trace_event
from agent.sandbox.models import RepoExecutionReport
from agent.tools.github_client import GitHubClient, RepoMeta

logger = logging.getLogger("exaai_adk.github_analyzer")

GITHUB_RESERVED_PATHS = {
    "sponsors",
    "settings",
    "trending",
    "features",
    "pricing",
    "explore",
    "about",
    "contact",
    "topics",
    "collections",
    "events",
    "orgs",
    "marketplace",
    "pulls",
    "issues",
}

# Second path segment in github.com/{owner}/{segment}/... that is not a repository name.
GITHUB_NON_REPO_SEGMENTS = frozenset(
    {
        "tree",
        "blob",
        "raw",
        "commits",
        "commit",
        "issues",
        "pull",
        "pulls",
        "wiki",
        "discussions",
        "actions",
        "projects",
        "packages",
        "settings",
        "releases",
        "tags",
        "stargazers",
        "network",
        "graphs",
        "security",
        "login",
        "signup",
    }
)

# Top programming languages and their standard file extensions
LANG_EXTENSIONS = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".jsx": "JavaScript",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".cpp": "C++",
    ".cc": "C++",
    ".c": "C",
    ".h": "C/C++ Header",
    ".cs": "C#",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".scala": "Scala",
    ".sh": "Shell",
    ".yml": "YAML",
    ".yaml": "YAML",
}


@dataclass
class RepoAnalysis:
    name: str
    url: str
    description: str | None
    languages: dict[str, float]  # Language name -> percentage of bytes
    stars: int
    is_fork: bool
    project_type: str  # "library", "web-app", "cli-tool", "data-pipeline", "unknown", etc.
    has_tests: bool
    has_ci: bool
    has_docs: bool
    has_docker: bool
    dependency_summary: str
    code_samples: list[str] = field(default_factory=list)
    commit_frequency: str = "stale"
    commit_quality: str = "basic"
    complexity_estimate: str = "simple"
    commits: list[dict[str, Any]] = field(default_factory=list)
    repo_type_tags: list[str] = field(default_factory=list)
    github_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GitHubAnalysis:
    username: str
    total_public_repos: int
    total_stars: int
    primary_languages: list[str]
    repo_analyses: list[RepoAnalysis]
    coding_style_summary: str
    overall_github_signal: str  # "strong", "moderate", "weak", "none"
    collaboration_summary: str = ""
    commit_hygiene: str = ""
    resume_github_repo_urls: list[str] = field(default_factory=list)
    discovered_github_repo_urls: list[str] = field(default_factory=list)
    selected_sandbox_repo_urls: list[str] = field(default_factory=list)
    sandbox_reports: list[dict[str, Any]] = field(default_factory=list)
    repo_selection_mode: str = "none"
    candidate_tags: list[str] = field(default_factory=list)
    github_metadata: dict[str, Any] = field(default_factory=dict)
    user_profile: dict[str, Any] = field(default_factory=dict)
    profile_readme: str = ""
    activity_timeline: dict[str, Any] = field(default_factory=dict)


def extract_github_owner_from_repo_url(url: str) -> str | None:
    """Owner login from a repository URL; None for profile-only or reserved paths."""
    normalized = normalize_github_repo_url(url)
    if not normalized:
        return None
    match = re.search(r"github\.com/([^/]+)/", normalized, re.IGNORECASE)
    if not match:
        return None
    return match.group(1)


def extract_github_owners_from_repo_urls(urls: list[str]) -> list[str]:
    """Owners from repo URLs in first-seen order, deduped case-insensitively."""
    seen: set[str] = set()
    owners: list[str] = []
    for url in urls:
        owner = extract_github_owner_from_repo_url(url)
        if not owner:
            continue
        key = owner.lower()
        if key in seen:
            continue
        seen.add(key)
        owners.append(owner)
    return owners


def resolve_github_username_from_repos(
    urls: list[str],
    *,
    owner_hint: str | None = None,
) -> str | None:
    """Pick one owner when multiple repo URLs agree; None if ambiguous."""
    from collections import Counter

    repo_urls = extract_github_repo_urls(urls)
    counts: Counter[str] = Counter()
    canonical: dict[str, str] = {}
    for url in repo_urls:
        owner = extract_github_owner_from_repo_url(url)
        if not owner:
            continue
        key = owner.lower()
        counts[key] += 1
        canonical.setdefault(key, owner)

    if not counts:
        return None

    hint = str(owner_hint or "").strip().lower()
    if hint and hint in counts:
        return canonical[hint]

    if len(counts) == 1:
        return canonical[next(iter(counts))]

    total = sum(counts.values())
    top_key, top_count = counts.most_common(1)[0]
    if total > 0 and (top_count / total) >= 0.8:
        return canonical[top_key]

    logger.info("ambiguous_github_owners owners=%s", dict(counts))
    return None


def extract_github_profile_username(urls: list[str]) -> str | None:
    """First explicit GitHub profile URL owner (not a repo path)."""
    for url in urls:
        if not url:
            continue
        profile = normalize_github_profile_url(url)
        if not profile:
            continue
        match = re.search(r"github\.com/([^/]+)$", profile, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_github_username(urls: list[str]) -> str | None:
    """Extract GitHub username from profile URLs, else consensus from repo URLs."""
    profile_user = extract_github_profile_username(urls)
    if profile_user:
        return profile_user
    return resolve_github_username_from_repos(urls)


def _owner_hint_from_state(state: dict[str, Any] | None) -> str | None:
    if not isinstance(state, dict):
        return None
    cached = str(state.get("github_username") or "").strip()
    if cached:
        return cached
    resume = state.get("resume_structured")
    if isinstance(resume, dict):
        for key in ("github_username", "github_handle"):
            value = str(resume.get(key) or "").strip()
            if value:
                return value.lstrip("@")
    return None


def resolve_github_username_with_source(
    state: dict[str, Any] | None = None,
    *,
    explicit_username: str | None = None,
    profile_urls: list[str] | None = None,
    discovered_repo_urls: list[str] | None = None,
) -> tuple[str | None, str | None]:
    """
    Resolve GitHub username and how it was inferred.

    Returns (username, source) where source is one of:
    explicit, cached, analysis, profile_url, repo_urls, or None.
    """
    explicit = str(explicit_username or "").strip()
    if explicit:
        return explicit, "explicit"

    if isinstance(state, dict):
        cached = str(state.get("github_username") or "").strip()
        if cached:
            return cached, "cached"
        github = state.get("github_repo_analyses")
        if isinstance(github, dict):
            from_analysis = str(github.get("username") or "").strip()
            if from_analysis:
                return from_analysis, "analysis"

    profile_candidates: list[str] = []
    repo_candidates: list[str] = []

    if isinstance(state, dict):
        profile_candidates.extend(str(url) for url in list(state.get("profile_urls") or []) if url)
        profile_candidates.extend(
            str(url) for url in list(state.get("discovered_profile_urls") or []) if url
        )
        repo_candidates.extend(
            str(url) for url in list(state.get("discovered_github_repo_urls") or []) if url
        )
        github = state.get("github_repo_analyses")
        if isinstance(github, dict):
            for key in ("resume_github_repo_urls", "selected_sandbox_repo_urls"):
                repo_candidates.extend(str(url) for url in list(github.get(key) or []) if url)
            repo_candidates.extend(
                str(url) for url in list(github.get("discovered_github_repo_urls") or []) if url
            )

    profile_candidates.extend(str(url) for url in list(profile_urls or []) if url)
    repo_candidates.extend(str(url) for url in list(discovered_repo_urls or []) if url)

    profile_user = extract_github_profile_username(profile_candidates)
    if profile_user:
        return profile_user, "profile_url"

    repo_url_lists = [
        repo_candidates,
        extract_github_repo_urls(profile_candidates),
        extract_github_repo_urls(profile_candidates + repo_candidates),
    ]
    merged_repos = merge_github_repo_urls(*repo_url_lists)
    owner_hint = _owner_hint_from_state(state)
    repo_user = resolve_github_username_from_repos(merged_repos, owner_hint=owner_hint)
    if repo_user:
        return repo_user, "repo_urls"

    return None, None


def resolve_github_username(
    state: dict[str, Any] | None = None,
    *,
    explicit_username: str | None = None,
    profile_urls: list[str] | None = None,
    discovered_repo_urls: list[str] | None = None,
) -> str | None:
    """Resolve GitHub username from session state and/or URL lists."""
    username, _source = resolve_github_username_with_source(
        state,
        explicit_username=explicit_username,
        profile_urls=profile_urls,
        discovered_repo_urls=discovered_repo_urls,
    )
    return username


def sync_github_identity(state: dict[str, Any]) -> str | None:
    """Propagate GitHub username and minimal analysis shell after portfolio discovery."""
    username, source = resolve_github_username_with_source(state)
    if not username:
        return None

    state["github_username"] = username
    if source:
        state["github_username_source"] = source
    github = state.get("github_repo_analyses")
    if not isinstance(github, dict):
        github = {}
    if not github.get("username"):
        github = {
            **github,
            "username": username,
            "repo_analyses": list(github.get("repo_analyses") or []),
            "candidate_tags": list(github.get("candidate_tags") or []),
        }
        state["github_repo_analyses"] = github
    return username


def ensure_minimal_github_shell_from_repos(
    state: dict[str, Any],
    repo_urls: list[str] | None = None,
) -> str | None:
    """Infer username from repo URLs and seed a minimal github_repo_analyses shell."""
    urls = list(repo_urls or [])
    if not urls:
        urls = merge_github_repo_urls(
            list(state.get("discovered_github_repo_urls") or []),
            extract_github_repo_urls(list(state.get("profile_urls") or [])),
            extract_github_repo_urls(list(state.get("discovered_profile_urls") or [])),
        )
        github = state.get("github_repo_analyses")
        if isinstance(github, dict):
            urls = merge_github_repo_urls(
                urls,
                list(github.get("resume_github_repo_urls") or []),
                list(github.get("discovered_github_repo_urls") or []),
            )
    if not urls:
        return None

    username = resolve_github_username_from_repos(urls, owner_hint=_owner_hint_from_state(state))
    if not username:
        return None

    state["github_username"] = username
    state.setdefault("github_username_source", "repo_urls")
    github = state.get("github_repo_analyses")
    if not isinstance(github, dict):
        github = {}
    resume_repos = list(github.get("resume_github_repo_urls") or [])
    if not resume_repos:
        resume_repos = extract_github_repo_urls(list(state.get("profile_urls") or []))
    state["github_repo_analyses"] = {
        **github,
        "username": username,
        "resume_github_repo_urls": resume_repos or urls,
        "discovered_github_repo_urls": list(
            github.get("discovered_github_repo_urls")
            or state.get("discovered_github_repo_urls")
            or []
        ),
        "repo_analyses": list(github.get("repo_analyses") or []),
        "candidate_tags": list(github.get("candidate_tags") or []),
    }
    return username


async def ensure_github_analysis_after_discovery(state: dict[str, Any]) -> bool:
    """Run GitHub API analysis when discovery resolves a username but prep did not."""
    sync_github_identity(state)
    username = resolve_github_username(state)
    if not username:
        return False

    github = state.get("github_repo_analyses")
    if isinstance(github, dict) and github.get("repo_analyses"):
        return True

    from agent.sandbox_gating import sandbox_mode_for_settings

    try:
        analysis = await analyze_github_repos(
            username=username,
            repo_urls=list(state.get("profile_urls") or []),
            discovered_repo_urls=list(state.get("discovered_github_repo_urls") or []),
            jd_structured=state.get("jd_structured") or {},
            sandbox_mode=sandbox_mode_for_settings(),
        )
        state["github_repo_analyses"] = analysis
        state["github_username"] = username
        trace_event(
            logger,
            "github_analysis_after_discovery",
            username=username,
            repo_count=len(analysis.get("repo_analyses") or []),
        )
        return True
    except Exception as exc:
        logger.warning("GitHub analysis after discovery failed for %s: %s", username, exc)
        return False


def normalize_github_profile_url(url: str) -> str | None:
    """Normalize a GitHub profile URL (``github.com/{user}``), excluding repo paths."""
    match = re.search(r"https?://(?:www\.)?github\.com/([^/?#]+)(?:/([^/?#]+))?", url or "")
    if not match:
        return None
    owner = match.group(1).strip()
    repo_segment = match.group(2)
    if repo_segment:
        return None
    if not owner or owner.lower() in GITHUB_RESERVED_PATHS:
        return None
    return f"https://github.com/{owner}"


def normalize_github_repo_url(url: str) -> str | None:
    """Normalize a GitHub repository URL, or return None for profile/non-repo URLs."""
    match = re.search(r"https?://(?:www\.)?github\.com/([^/?#]+)/([^/?#]+)", url or "")
    if not match:
        return None

    owner = match.group(1).strip()
    repo = match.group(2).strip().removesuffix(".git")
    if (
        not owner
        or not repo
        or owner.lower() in GITHUB_RESERVED_PATHS
        or repo.lower() in GITHUB_NON_REPO_SEGMENTS
    ):
        return None
    return f"https://github.com/{owner}/{repo}"


def extract_github_repo_urls(urls: list[str]) -> list[str]:
    """Extract unique resume-mentioned GitHub repository URLs in resume order."""
    seen: set[str] = set()
    repo_urls: list[str] = []
    for url in urls:
        normalized = normalize_github_repo_url(url)
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            repo_urls.append(normalized)
    return repo_urls


def merge_github_repo_urls(*repo_url_lists: list[str]) -> list[str]:
    """Merge multiple repo URL lists preserving first-seen order."""
    seen: set[str] = set()
    merged: list[str] = []
    for urls in repo_url_lists:
        for raw in urls:
            normalized = normalize_github_repo_url(raw)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
    return merged


def _repo_key_from_url(url: str) -> tuple[str, str] | None:
    normalized = normalize_github_repo_url(url)
    if normalized is None:
        return None
    match = re.search(r"github\.com/([^/]+)/([^/]+)$", normalized)
    if not match:
        return None
    owner = match.group(1)
    repo = match.group(2)
    return owner.lower(), repo.lower()


def _resume_repo_cap(settings: Any) -> int:
    configured = int(getattr(settings, "sandbox_max_resume_repos", 12) or 12)
    return max(1, min(configured, 12))


def _sandbox_batch_wait_seconds(settings: Any, repo_count: int) -> float:
    """Wall-clock budget for a parallel sandbox batch (grows with repo count)."""
    base = float(getattr(settings, "sandbox_wait_seconds", 45.0) or 0)
    if base <= 0 or repo_count <= 0:
        return base
    if repo_count == 1:
        return base
    # Extra repos run in parallel but stagger Cloud Run job completion.
    return base + (repo_count - 1) * (base * 0.35)


def align_sandbox_reports_with_urls(
    urls: list[str],
    reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ensure one report per selected URL, preserving resume order."""
    by_url: dict[str, dict[str, Any]] = {}
    for report in reports:
        if not isinstance(report, dict):
            continue
        raw_url = str(report.get("url") or "").strip()
        if not raw_url:
            continue
        canonical = normalize_github_repo_url(raw_url) or raw_url.rstrip("/").removesuffix(".git")
        by_url[canonical] = report
        by_url[raw_url] = report

    aligned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in urls:
        url = str(raw or "").strip()
        if not url:
            continue
        canonical = normalize_github_repo_url(url) or url.rstrip("/").removesuffix(".git")
        report = by_url.get(canonical) or by_url.get(url)
        if report is not None and canonical not in seen:
            aligned.append(report)
            seen.add(canonical)

    if aligned:
        return aligned
    return [report for report in reports if isinstance(report, dict)]


async def _resolve_resume_repos_for_analysis(
    *,
    client: GitHubClient,
    all_repos: list[RepoMeta],
    resume_repo_urls: list[str],
    settings: Any,
) -> list[RepoMeta]:
    """Resolve every resume-listed repo URL to RepoMeta (profile list or direct API lookup)."""
    by_key = {(repo.owner.lower(), repo.name.lower()): repo for repo in all_repos}
    selected: list[RepoMeta] = []
    seen: set[tuple[str, str]] = set()

    for url in resume_repo_urls[: _resume_repo_cap(settings)]:
        key = _repo_key_from_url(url)
        if not key or key in seen:
            continue
        seen.add(key)
        if key in by_key:
            selected.append(by_key[key])
            continue
        owner, name = key
        repo = await client.get_repo_meta(owner, name)
        if repo is not None:
            selected.append(repo)
    return selected


def _select_static_repos(
    *,
    resolved_resume_repos: list[RepoMeta],
    ranked_repos: list[RepoMeta],
    resume_repo_urls: list[str],
    settings: Any,
) -> tuple[list[RepoMeta], str]:
    """Select repos for static GitHub API analysis."""
    if resume_repo_urls and resolved_resume_repos:
        return resolved_resume_repos, "resume_repos"

    limit = getattr(settings, "max_repos_to_analyze", 3)
    return ranked_repos[:limit], "ranked_profile_repos"


def _select_sandbox_repo_urls(
    *,
    ranked_repos: list[RepoMeta],
    resume_repo_urls: list[str],
    settings: Any,
) -> tuple[list[str], str]:
    """Select repos for sandbox cloning — all resume repo links, in resume order."""
    if resume_repo_urls:
        return resume_repo_urls[: _resume_repo_cap(settings)], "resume_repos"

    limit = getattr(settings, "sandbox_max_profile_repos", 2)
    return [repo.url for repo in ranked_repos[:limit]], "ranked_profile_repos"


def _repo_name_from_url(repo_url: str) -> str:
    normalized = normalize_github_repo_url(repo_url) or repo_url.rstrip("/")
    parts = normalized.removesuffix(".git").split("/")
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return normalized or "unknown"


def _sandbox_skip_report(
    *,
    url: str,
    settings: Any,
    reason: str,
    summary: str,
    timed_out: bool = False,
) -> dict[str, Any]:
    return RepoExecutionReport(
        repo=_repo_name_from_url(url),
        url=url,
        provider=str(getattr(settings, "sandbox_provider", "unknown")),
        clone_ok=False,
        skipped_reason=reason,
        summary=summary,
        timed_out=timed_out,
    ).compact()


async def _evaluate_sandbox_repos(
    repo_urls: list[str],
    settings: Any,
    *,
    file_focus_by_url: dict[str, dict[str, Any]] | None = None,
    _retry_pass: int = 0,
) -> list[dict[str, Any]]:
    """Evaluate selected repositories in parallel through the configured sandbox provider."""
    if not repo_urls:
        return []

    from agent.sandbox.providers import create_sandbox_provider

    try:
        provider = create_sandbox_provider()
    except Exception as exc:
        logger.warning("Unable to create sandbox provider: %s", exc)
        return [
            _sandbox_skip_report(
                url=url,
                settings=settings,
                reason=f"Sandbox provider unavailable: {exc}",
                summary="Sandbox evaluation was skipped because provider setup failed.",
            )
            for url in repo_urls
        ]

    focus_map = file_focus_by_url if isinstance(file_focus_by_url, dict) else {}

    async def evaluate(url: str) -> dict[str, Any]:
        repo_name = _repo_name_from_url(url)
        try:
            report = await provider.evaluate_repo(
                repo_url=url,
                repo_name=repo_name,
                commands=[],
                file_focus=focus_map.get(url),
            )
            return report.compact()
        except Exception as exc:
            logger.warning("Sandbox evaluation failed for %s: %s", url, exc)
            return _sandbox_skip_report(
                url=url,
                settings=settings,
                reason=f"Sandbox evaluation failed: {exc}",
                summary="Sandbox evaluation failed before a report was produced.",
            )

    batch_wait = _sandbox_batch_wait_seconds(settings, len(repo_urls))
    tasks = {asyncio.create_task(evaluate(url)): url for url in repo_urls}
    results_by_url: dict[str, dict[str, Any]] = {}

    if batch_wait > 0:
        done, pending = await asyncio.wait(set(tasks.keys()), timeout=batch_wait)
        for task in done:
            url = tasks[task]
            try:
                results_by_url[url] = task.result()
            except Exception as exc:
                logger.warning("Sandbox task failed for %s: %s", url, exc)
                results_by_url[url] = _sandbox_skip_report(
                    url=url,
                    settings=settings,
                    reason=f"Sandbox evaluation failed: {exc}",
                    summary="Sandbox evaluation failed before a report was produced.",
                )
        if pending:
            logger.warning(
                "Sandbox batch timed out after %ss with %s/%s repo(s) still running",
                batch_wait,
                len(pending),
                len(repo_urls),
            )
            for task in pending:
                task.cancel()
                url = tasks[task]
                results_by_url[url] = _sandbox_skip_report(
                    url=url,
                    settings=settings,
                    reason=f"Sandbox wait budget exceeded after {batch_wait:.0f}s.",
                    summary=(
                        "Sandbox evaluation did not finish within the screening wait budget; "
                        "static GitHub evidence was used instead."
                    ),
                    timed_out=True,
                )
    else:
        gathered = await asyncio.gather(*(evaluate(url) for url in repo_urls))
        results_by_url = dict(zip(repo_urls, gathered))

    timed_out_urls = [
        url for url in repo_urls if results_by_url.get(url, {}).get("timed_out") is True
    ]
    if timed_out_urls and _retry_pass < 1:
        logger.info("Retrying %s sandbox repo(s) that timed out on first pass", len(timed_out_urls))
        retry_reports = await _evaluate_sandbox_repos(
            timed_out_urls,
            settings,
            file_focus_by_url=focus_map,
            _retry_pass=_retry_pass + 1,
        )
        for url, report in zip(timed_out_urls, retry_reports):
            results_by_url[url] = report

    return [results_by_url[url] for url in repo_urls]


def get_jd_keywords(jd_structured: dict[str, Any] | None) -> set[str]:
    """Extract technology keywords from the job description for ranking and classification."""
    return _get_jd_keywords(jd_structured)


def _get_jd_keywords(jd_structured: dict[str, Any] | None) -> set[str]:
    """Extract technology keywords from the job description for ranking."""
    keywords = set()
    if not jd_structured:
        return keywords

    # Add words from job title, must have, nice to have
    title = jd_structured.get("job_title") or ""
    for w in re.findall(r"\w+", title.lower()):
        if len(w) > 2:
            keywords.add(w)

    for item in jd_structured.get("must_have") or []:
        for w in re.findall(r"\w+", item.lower()):
            if len(w) > 2:
                keywords.add(w)

    for item in jd_structured.get("nice_to_have") or []:
        for w in re.findall(r"\w+", item.lower()):
            if len(w) > 2:
                keywords.add(w)

    # Common programming languages/tech
    tech_list = [
        "python",
        "javascript",
        "typescript",
        "golang",
        "go",
        "rust",
        "java",
        "cpp",
        "c++",
        "csharp",
        "c#",
        "ruby",
        "php",
        "swift",
        "kotlin",
        "scala",
        "docker",
        "kubernetes",
        "aws",
        "gcp",
        "azure",
        "fastapi",
        "flask",
        "django",
        "react",
        "vue",
        "angular",
        "node",
        "express",
        "nextjs",
        "next.js",
        "pytorch",
        "tensorflow",
    ]
    for tech in tech_list:
        # Check if they are mentioned in JD
        jd_text_lower = json.dumps(jd_structured).lower()
        if re.search(rf"\b{re.escape(tech)}\b", jd_text_lower):
            keywords.add(tech)

    return keywords


def _classify_candidate_tags(
    jd_structured: dict[str, Any] | None,
    repos_data: list[dict[str, Any]],
) -> list[str]:
    text = json.dumps(jd_structured or {}).lower()
    text += " " + json.dumps(repos_data).lower()
    mapping = {
        "frontend_engineer": ("react", "next", "frontend", "typescript"),
        "backend_engineer": ("fastapi", "flask", "express", "backend", "api"),
        "fullstack_engineer": ("fullstack", "frontend", "backend"),
        "ml_engineer": ("ml", "training", "pytorch", "tensorflow"),
        "ai_engineer": (
            "langchain", "llm", "rag", "agent", "ai", "embedding", "retrieval", "vector",
        ),
        "cybersecurity_engineer": ("security", "siem", "ctf", "vulnerability", "threat"),
        "data_engineer": ("airflow", "pipeline", "warehouse", "etl", "data", "embedding"),
        "devops_platform_engineer": ("terraform", "helm", "kubernetes", "platform", "devops"),
        "general_software_engineer": ("python", "java", "go", "javascript"),
    }
    tags = [tag for tag, needles in mapping.items() if any(needle in text for needle in needles)]
    if not tags:
        tags.append("general_software_engineer")
    return sorted(dict.fromkeys(tags))


def _build_repo_github_metadata(repo: RepoMeta) -> dict[str, Any]:
    return {
        "stars": repo.stars,
        "forks": repo.forks,
        "watchers": None,
        "default_branch": repo.default_branch,
        "archived": None,
        "topics": repo.topics,
        "created_at": repo.created_at or None,
        "last_push_at": repo.updated_at,
        "license_present": None,
        "has_wiki": None,
        "has_pages": None,
        "open_issue_ratio": None,
        "open_pr_ratio": None,
        "release_count": None,
        "contributors_count": None,
    }


def _user_profile_to_dict(user: Any) -> dict[str, Any]:
    if user is None:
        return {}
    return {
        "login": getattr(user, "login", "") or "",
        "html_url": getattr(user, "html_url", "") or "",
        "created_at": getattr(user, "created_at", "") or "",
        "bio": getattr(user, "bio", None),
        "blog": getattr(user, "blog", None),
        "twitter_username": getattr(user, "twitter_username", None),
        "email": getattr(user, "email", None),
    }


def _parse_iso_datetime(value: str) -> datetime | None:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    try:
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _build_activity_timeline(
    *,
    resume_repo_urls: list[str],
    repo_analyses: list[RepoAnalysis],
    all_repos: list[RepoMeta],
) -> dict[str, Any]:
    """Earliest/latest activity among resume-listed GitHub repos."""
    resume_keys = {
        url.rstrip("/").lower()
        for url in resume_repo_urls
        if isinstance(url, str) and url.strip()
    }
    resume_names = {
        url.rstrip("/").split("/")[-1].lower()
        for url in resume_repo_urls
        if isinstance(url, str) and "/" in url
    }

    activity_dates: list[datetime] = []
    matched = 0
    for analysis in repo_analyses:
        analysis_url = str(analysis.url or "").rstrip("/").lower()
        analysis_name = analysis_url.split("/")[-1] if analysis_url else ""
        if resume_keys and analysis_url not in resume_keys and analysis_name not in resume_names:
            continue
        matched += 1
        meta_created = analysis.github_metadata.get("created_at")
        if isinstance(meta_created, str):
            parsed = _parse_iso_datetime(meta_created)
            if parsed:
                activity_dates.append(parsed)
        for commit in analysis.commits or []:
            if not isinstance(commit, dict):
                continue
            parsed = _parse_iso_datetime(str(commit.get("date") or ""))
            if parsed:
                activity_dates.append(parsed)

    if not activity_dates:
        for repo in all_repos:
            repo_url = str(repo.url or "").rstrip("/").lower()
            repo_name = repo.name.lower()
            if resume_keys and repo_url not in resume_keys and repo_name not in resume_names:
                continue
            matched += 1
            parsed = _parse_iso_datetime(repo.created_at)
            if parsed:
                activity_dates.append(parsed)
            parsed = _parse_iso_datetime(repo.updated_at)
            if parsed:
                activity_dates.append(parsed)

    earliest = min(activity_dates) if activity_dates else None
    latest = max(activity_dates) if activity_dates else None
    return {
        "earliest_activity_at": earliest.isoformat().replace("+00:00", "Z") if earliest else None,
        "latest_activity_at": latest.isoformat().replace("+00:00", "Z") if latest else None,
        "resume_repo_count": len(resume_repo_urls),
        "matched_resume_repos": matched,
    }


def _score_repo_relevance(repo: RepoMeta, jd_keywords: set[str]) -> float:
    """Score repository based on its relevance to the job description."""
    score = 0.0
    name_lower = repo.name.lower()
    desc_lower = (repo.description or "").lower()

    if repo.is_fork:
        score -= 10.0

    # 1. Primary language matches JD keyword
    if repo.language and repo.language.lower() in jd_keywords:
        score += 15.0

    # 2. Topic matches JD keyword
    for topic in repo.topics:
        if topic.lower() in jd_keywords:
            score += 5.0

    # 3. Name or description matches JD keyword
    for kw in jd_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", name_lower):
            score += 10.0
        if re.search(rf"\b{re.escape(kw)}\b", desc_lower):
            score += 2.0

    # 4. Popularity (stars)
    score += min(repo.stars * 0.1, 10.0)

    return score


def _analyze_collaboration(username: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse user events to compute collaboration and open-source metrics."""
    pr_created = 0
    pr_reviewed = 0
    external_repos = set()

    for event in events:
        e_type = event.get("type")
        repo_info = event.get("repo", {})
        repo_name = repo_info.get("name", "")

        owner = ""
        if repo_name and "/" in repo_name:
            owner = repo_name.split("/")[0]

        # Check if it's an external repository
        is_external = bool(owner and owner.lower() != username.lower())

        if e_type == "PullRequestEvent":
            action = event.get("payload", {}).get("action")
            if action == "opened":
                pr_created += 1
                if is_external:
                    external_repos.add(repo_name)
        elif e_type == "PullRequestReviewEvent":
            pr_reviewed += 1
            if is_external:
                external_repos.add(repo_name)
        elif e_type in ("PushEvent", "IssueCommentEvent", "IssuesEvent"):
            if is_external and repo_name:
                external_repos.add(repo_name)

    ext_list = sorted(list(external_repos))
    summary_parts = []
    summary_parts.append(
        f"In recent activity: opened {pr_created} PRs, reviewed {pr_reviewed} PRs."
    )
    if ext_list:
        summary_parts.append(f"Contributed to external repositories: {', '.join(ext_list)}.")
    else:
        summary_parts.append("No external repository contributions detected in recent events.")

    return {
        "pull_requests_created": pr_created,
        "pull_requests_reviewed": pr_reviewed,
        "external_contributions": ext_list,
        "summary": " ".join(summary_parts),
    }


def _score_path_relevance(path: str, jd_keywords: set[str]) -> float:
    """Score a file path based on its keyword and path relevance to JD."""
    score = 0.0
    path_lower = path.lower()

    # Prioritize paths inside common source directories
    if any(x in path_lower for x in {"src/", "app/", "lib/", "pkg/"}):
        score += 5.0

    # Split path by delimiters to look for keyword matches
    parts = re.split(r"[/\-_.]", path_lower)
    for part in parts:
        if part in jd_keywords:
            score += 10.0

    # Specific common programming patterns / component names matches
    patterns = {
        "auth": 3.0,
        "db": 3.0,
        "sql": 3.0,
        "model": 2.0,
        "controller": 2.0,
        "route": 2.0,
        "handler": 2.0,
        "service": 2.0,
        "client": 2.0,
        "utils": 1.0,
        "helper": 1.0,
    }
    for pat, weight in patterns.items():
        if pat in path_lower:
            score += weight

    return score


def _extract_relevant_snippet(content: str, jd_keywords: set[str], max_chars: int = 1500) -> str:
    """Locate matching JD keywords in content and extract a relevant snippet."""
    if not content:
        return ""

    # If content is short, return the whole thing
    if len(content) <= max_chars:
        return content

    lines = content.splitlines()
    best_line_idx = -1
    max_matches = 0

    # Find the line with the highest number of keyword matches
    for idx, line in enumerate(lines):
        line_lower = line.lower()
        matches = sum(1 for kw in jd_keywords if kw in line_lower)
        if matches > max_matches:
            max_matches = matches
            best_line_idx = idx

    if best_line_idx == -1:
        # If no keyword matches, fallback to the first match of any keyword
        for idx, line in enumerate(lines):
            line_lower = line.lower()
            if any(kw in line_lower for kw in jd_keywords):
                best_line_idx = idx
                break

    if best_line_idx == -1:
        # Fallback: return the beginning of the file
        return content[:max_chars]

    # Center the snippet around the best matching line
    # Try to start up to 3 lines before the matching line
    start_idx = max(0, best_line_idx - 3)

    snippet_lines = []
    current_len = 0
    for line in lines[start_idx:]:
        if current_len + len(line) + 1 > max_chars:
            if not snippet_lines:
                # If even one line is longer than max_chars, truncate it
                return line[:max_chars]
            break
        snippet_lines.append(line)
        current_len += len(line) + 1

    return "\n".join(snippet_lines)


def _parse_dependencies_from_code(path: str, content: str) -> set[str]:
    """Parse imports or require statements to extract used libraries."""
    deps = set()

    if path.endswith(".py"):
        # Matches: import package, import package.module, from package import module
        for match in re.finditer(r"^\s*(?:import|from)\s+([a-zA-Z0-9_]+)", content, re.MULTILINE):
            pkg = match.group(1)
            if pkg not in {
                "os",
                "sys",
                "time",
                "json",
                "math",
                "re",
                "datetime",
                "typing",
                "collections",
                "hashlib",
                "shutil",
                "tempfile",
            }:
                deps.add(pkg)
    elif path.endswith((".js", ".jsx", ".ts", ".tsx")):
        # Matches: import package from 'package', require('package')
        for match in re.finditer(r"from\s+['\"]([^'\"]+)['\"]", content):
            pkg = match.group(1).split("/")[0]
            if not pkg.startswith("."):
                deps.add(pkg)
        for match in re.finditer(r"require\(\s*['\"]([^'\"]+)['\"]", content):
            pkg = match.group(1).split("/")[0]
            if not pkg.startswith("."):
                deps.add(pkg)
    elif path.endswith(".go"):
        # Matches: import "package" or import ( ... )
        single_imports = re.finditer(r'import\s+"([^"]+)"', content)
        for m in single_imports:
            pkg = m.group(1).split("/")[-1]
            deps.add(pkg)
        block_imports = re.search(r"import\s+\((.*?)\)", content, re.DOTALL)
        if block_imports:
            for line in block_imports.group(1).splitlines():
                line = line.strip().strip('"')
                if line:
                    pkg = line.split("/")[-1]
                    deps.add(pkg)
    return deps


def _extract_dependencies_from_manifest(path: str, content: str) -> set[str]:
    """Parse package/dependency manifests (e.g. package.json, pyproject.toml)."""
    deps = set()
    filename = path.split("/")[-1]

    try:
        if filename == "package.json":
            data = json.loads(content)
            for k in ("dependencies", "devDependencies"):
                if k in data and isinstance(data[k], dict):
                    deps.update(data[k].keys())
        elif filename == "pyproject.toml":
            # PEP 621 dependencies array
            matches = re.findall(r"dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL)
            for m in matches:
                for dep in re.findall(r'"([^"]+)"|\'([^\']+)\'', m):
                    dep_name = dep[0] or dep[1]
                    name_clean = re.split(r"[<>=~!]", dep_name)[0].strip()
                    if name_clean:
                        deps.add(name_clean)
            # Poetry/Pipenv key-value style under dependency headers
            in_deps_section = False
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    header = line[1:-1].lower()
                    if "dependencies" in header or "dev-dependencies" in header:
                        in_deps_section = True
                    else:
                        in_deps_section = False
                    continue
                if in_deps_section and "=" in line:
                    dep_name = line.split("=")[0].strip()
                    if dep_name and dep_name.lower() != "python":
                        deps.add(dep_name)
        elif filename == "requirements.txt":
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith(("#", "-r", "-e")):
                    name_clean = re.split(r"[<>=~!]", line)[0].strip()
                    if name_clean:
                        deps.add(name_clean)
        elif filename == "Cargo.toml":
            in_deps = False
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("[dependencies]") or line.startswith("[dev-dependencies]"):
                    in_deps = True
                    continue
                elif line.startswith("[") and in_deps:
                    in_deps = False
                if in_deps and "=" in line:
                    dep_name = line.split("=")[0].strip()
                    if dep_name:
                        deps.add(dep_name)
        elif filename == "go.mod":
            in_require = False
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("require ("):
                    in_require = True
                    continue
                elif line == ")":
                    in_require = False
                if in_require:
                    parts = line.split()
                    if parts:
                        pkg = parts[0].split("/")[-1]
                        deps.add(pkg)
                elif line.startswith("require "):
                    parts = line.split()
                    if len(parts) > 1:
                        pkg = parts[1].split("/")[-1]
                        deps.add(pkg)
    except Exception as e:
        logger.warning(f"Error parsing manifest {path}: {e}")

    return deps


class GitHubSummaryResponse(pydantic.BaseModel):
    coding_style_summary: str
    overall_github_signal: Literal["strong", "moderate", "weak", "none"]
    collaboration_style: str
    commit_hygiene: str


async def _generate_coding_style_summary(
    username: str, repos_data: list[dict[str, Any]], collaboration_summary: str, settings: Any
) -> tuple[str, str, str, str]:
    """Call the LLM to summarize the candidate's coding style and depth."""
    # Build a compact summary of repos for prompt
    repos_summary = []
    for r in repos_data:
        commits_list = r.get("commits") or []
        commits_str = "\n".join(f"  - {c['message']} ({c['date'][:10]})" for c in commits_list)
        repos_summary.append(
            f"Repo: {r['name']}\n"
            f"Description: {r['description']}\n"
            f"Languages: {r['languages']}\n"
            f"Stars: {r['stars']}\n"
            f"Project Type: {r['project_type']}\n"
            f"Maturity Signals: has_tests={r['has_tests']}, "
            f"has_ci={r['has_ci']}, has_docs={r['has_docs']}, "
            f"has_docker={r['has_docker']}\n"
            f"Dependencies: {r['dependency_summary']}\n"
            f"Commit Frequency: {r['commit_frequency']}, "
            f"Commit Quality: {r['commit_quality']}, "
            f"Complexity: {r['complexity_estimate']}\n"
            f"Recent Commits:\n{commits_str}\n"
            f"Code Samples (excerpts):\n" + "\n---\n".join(r["code_samples"])[:2500]
        )

    repos_summary_str = "\n\n====================\n\n".join(repos_summary)

    prompt = (
        f"You are an expert technical recruiter and software engineer.\n"
        f"Evaluate the candidate's GitHub repositories, coding style, "
        f"code quality, commit hygiene, and collaboration signals.\n\n"
        f"Candidate Username: {username}\n\n"
        f"collaboration_summary (calculated from recent events):\n"
        f"{collaboration_summary}\n\n"
        f"Candidate Repositories Data:\n"
        f"{repos_summary_str}\n\n"
        f"Please generate a professional, objective analysis of the candidate's coding and "
        f"software engineering practices.\n"
        f"Format your output as a JSON object matching this schema:\n"
        f"{{\n"
        f'  "coding_style_summary": "A concise (2-3 sentences) summary of their '
        f"coding style, best practices (tests, CI/CD, documentation), framework usage, "
        f'and structural design.",\n'
        f'  "collaboration_style": "A concise (1-2 sentences) evaluation of their '
        f"collaboration style and open-source involvement based on the "
        f'collaboration summary.",\n'
        f'  "commit_hygiene": "A concise (1-2 sentences) evaluation of their commit '
        f"message quality, formatting (e.g. conventional commits), descriptive clarity, "
        f'and rigor based on their recent commit messages.",\n'
        f'  "overall_github_signal": "strong" | "moderate" | "weak" | "none"\n'
        f"}}\n\n"
        f"Ensure overall_github_signal corresponds to:\n"
        f'- "strong": Active development, clean structure, tests, CI/CD, documentation, '
        f"complex logic.\n"
        f'- "moderate": Good code samples but may lack testing, CI/CD, or recent activity.\n'
        f'- "weak": Mostly simple scripts, forks without contributions, or low code quality.\n'
        f'- "none": No code samples or repositories available.\n\n'
        f"JSON Response:"
    )

    from agent.llm_client import LITELLM_PROVIDERS, complete_json_for_provider, resolve_llm_provider

    provider = resolve_llm_provider(settings)

    try:
        if provider in LITELLM_PROVIDERS:
            data = complete_json_for_provider(
                prompt,
                settings=settings,
                provider=provider,
                schema_name="github_summary",
                schema={
                    "type": "object",
                    "properties": {
                        "coding_style_summary": {"type": "string"},
                        "overall_github_signal": {
                            "type": "string",
                            "enum": ["strong", "moderate", "weak", "none"],
                        },
                        "collaboration_style": {"type": "string"},
                        "commit_hygiene": {"type": "string"},
                    },
                    "required": [
                        "coding_style_summary",
                        "overall_github_signal",
                        "collaboration_style",
                        "commit_hygiene",
                    ],
                },
                max_tokens=1000,
            )
            return (
                data.get("coding_style_summary", ""),
                data.get("overall_github_signal", "none"),
                data.get("collaboration_style", ""),
                data.get("commit_hygiene", ""),
            )
        else:
            import time

            from google.genai import types
            from google.genai.errors import APIError, ServerError

            from agent.llm_client import create_genai_client

            client = create_genai_client(settings)
            max_retries = 3
            delay = 1.5
            response = None

            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=settings.gemini_model_id,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_json_schema=GitHubSummaryResponse.model_json_schema(),
                            temperature=0.1,
                            max_output_tokens=1000,
                        ),
                    )
                    break
                except (APIError, ServerError) as e:
                    status_code = getattr(e, "code", getattr(e, "status_code", None))
                    if (
                        status_code in (503, 429) or "503" in str(e) or "429" in str(e)
                    ) and attempt < max_retries - 1:
                        logger.warning(
                            f"Gemini API returned transient error during GitHub summary "
                            f"(attempt {attempt + 1}/{max_retries}). "
                            f"Retrying in {delay}s... Error: {e}"
                        )
                        time.sleep(delay)
                        delay *= 2.0
                    else:
                        raise

            if response is None:
                raise RuntimeError("Gemini call failed with no response")
            from agent.llm_client import increment_llm_call_count

            increment_llm_call_count(
                model=settings.gemini_model_id,
                source="github_coding_summary",
            )
            text = response.text or ""
            data = json.loads(text)
            return (
                data.get("coding_style_summary", ""),
                data.get("overall_github_signal", "none"),
                data.get("collaboration_style", ""),
                data.get("commit_hygiene", ""),
            )

    except Exception as e:
        logger.error(f"Failed to generate coding style summary via LLM: {e}")
        # Return fallback heuristic summary
        langs = set()
        for r in repos_data:
            langs.update(r["languages"].keys())
        summary_langs = ", ".join(langs)
        summary_repos = ", ".join(r["name"] for r in repos_data)
        summary = (
            f"Candidate has repositories in {summary_langs}. "
            f"Repos show projects like {summary_repos}."
        )
        return (
            summary,
            "moderate" if repos_data else "none",
            collaboration_summary,
            "Commit messages analyzed heuristically.",
        )


def _generate_coding_style_summary_heuristic(
    username: str,
    repos_data: list[dict[str, Any]],
    collaboration_summary: str,
) -> tuple[str, str, str, str]:
    """Generate a heuristic summary of the candidate's coding style and depth
    without an LLM call.
    """
    if not repos_data:
        return (
            f"No public repositories analyzed for GitHub user {username}.",
            "none",
            collaboration_summary,
            "No commits to analyze.",
        )

    langs = set()
    for r in repos_data:
        langs.update(r.get("languages", {}).keys())

    # Build coding style summary
    repo_names = [r["name"] for r in repos_data]
    lang_str = ", ".join(sorted(langs))
    coding_style_summary = (
        f"Candidate is active in {lang_str} across repositories including {', '.join(repo_names)}. "
    )

    # Check for engineering maturity indicators
    has_tests = any(r.get("has_tests") for r in repos_data)
    has_ci = any(r.get("has_ci") for r in repos_data)
    has_docs = any(r.get("has_docs") for r in repos_data)
    has_docker = any(r.get("has_docker") for r in repos_data)

    maturity_signals = []
    if has_tests:
        maturity_signals.append("testing")
    if has_ci:
        maturity_signals.append("CI/CD workflows")
    if has_docs:
        maturity_signals.append("documentation")
    if has_docker:
        maturity_signals.append("containerization")

    if maturity_signals:
        coding_style_summary += f"Codebases show evidence of {', '.join(maturity_signals)}."
    else:
        coding_style_summary += (
            "Codebases contain source files without explicit tests or CI configurations."
        )

    # Determine overall signal
    overall_github_signal = "weak"
    if repos_data:
        overall_github_signal = "moderate"
        # If they have tests/CI or highly-starred repositories or multiple complex repos
        high_quality_repos = sum(
            1
            for r in repos_data
            if r.get("has_tests")
            or r.get("has_ci")
            or r.get("stars", 0) >= 5
            or r.get("complexity_estimate") == "complex"
        )
        if high_quality_repos >= 1:
            overall_github_signal = "strong"

    # Commit hygiene
    commit_qualities = [r.get("commit_quality", "basic") for r in repos_data]
    if "descriptive" in commit_qualities:
        commit_hygiene = "Commit history shows descriptive, detailed commit messages."
    elif "poor" in commit_qualities:
        commit_hygiene = "Commit messages contain generic or short descriptions."
    else:
        commit_hygiene = "Commit messages are basic and functional."

    return coding_style_summary, overall_github_signal, collaboration_summary, commit_hygiene


async def _analyze_single_repo(
    repo: RepoMeta,
    client: GitHubClient,
    jd_keywords: set[str],
    settings: Any,
    remaining_char_budget_container: list[int],
) -> RepoAnalysis:
    """Analyze a single repository concurrently.

    ``remaining_char_budget_container`` is a mutable single-element list used
    to track the shared character budget across concurrent repo analyses.
    Because file-content fetching is sequential within a repo and repos are
    gathered concurrently, there is a minor race on the budget, but this is
    acceptable — it's a soft cap, not a hard limit.
    """
    logger.info(f"Analyzing repository {repo.owner}/{repo.name}")

    # --- Parallel fetch: languages, tree, commits ---
    raw_languages, tree, commits = await asyncio.gather(
        client.get_repo_languages(repo.owner, repo.name),
        client.get_repo_tree(repo.owner, repo.name, branch=repo.default_branch),
        client.get_recent_commits(repo.owner, repo.name, limit=10),
    )

    # Process languages
    total_bytes = sum(raw_languages.values())
    languages_pct = {}
    if total_bytes > 0:
        languages_pct = {k: round((v / total_bytes) * 100.0, 1) for k, v in raw_languages.items()}

    # Determine signals from paths
    file_paths = [entry.path for entry in tree]
    has_tests = any(
        "test" in p.lower()
        or p.endswith(("_test.py", ".test.js", ".test.ts", "Spec.scala", "Test.java"))
        for p in file_paths
    )
    has_ci = any(
        p.startswith(".github/workflows/")
        or p.endswith((".gitlab-ci.yml", "travis.yml", "circle.yml"))
        for p in file_paths
    )
    has_docs = any("doc" in p.lower() or p.endswith((".md", ".rst", ".txt")) for p in file_paths)
    has_docker = any(
        "docker" in p.lower()
        or p.endswith(("Dockerfile", "docker-compose.yml", "docker-compose.yaml"))
        for p in file_paths
    )

    # Estimate complexity
    code_file_count = sum(1 for p in file_paths if p.endswith(tuple(LANG_EXTENSIONS.keys())))
    if code_file_count > 50:
        complexity = "complex"
    elif code_file_count > 10:
        complexity = "moderate"
    else:
        complexity = "simple"

    # Determine project type
    project_type = "unknown"
    if has_docker:
        project_type = "dockerized-app"
    elif any("components" in p.lower() or "views" in p.lower() for p in file_paths):
        project_type = "web-app"
    elif any(
        p.endswith(("setup.py", "pyproject.toml", "Cargo.toml", "go.mod", "pom.xml"))
        for p in file_paths
    ):
        project_type = "library"
    elif code_file_count > 0:
        project_type = "codebase"

    # Commit pattern analysis
    commit_frequency = "stale"
    commit_quality = "basic"

    if commits:
        from datetime import datetime

        try:
            latest_date_str = commits[0].date.replace("Z", "")
            latest_date = datetime.fromisoformat(latest_date_str)
            delta_days = (datetime.now(UTC).replace(tzinfo=None) - latest_date).days
            if delta_days <= 30:
                commit_frequency = "active"
            elif delta_days <= 180:
                commit_frequency = "moderate"
        except Exception:
            pass

        messages = [c.message.lower() for c in commits]
        generic_count = sum(
            1
            for m in messages
            if any(
                x in m for x in {"update", "fix", "wip", "commit", "temp", "test", "add", "remove"}
            )
            and len(m.strip()) < 10
        )
        if generic_count > 7:
            commit_quality = "poor"
        elif generic_count > 3:
            commit_quality = "basic"
        else:
            commit_quality = "descriptive"

    # Read key files & dependency extraction
    manifest_files = ["package.json", "pyproject.toml", "requirements.txt", "Cargo.toml", "go.mod"]
    candidate_files: list[str] = []
    readme_file: str | None = None

    for path in file_paths:
        name_lower = path.split("/")[-1].lower()
        if name_lower in {"readme.md", "readme.txt", "readme"}:
            readme_file = path
            break

    if readme_file:
        candidate_files.append(readme_file)

    for path in file_paths:
        name_lower = path.split("/")[-1].lower()
        if name_lower in manifest_files:
            candidate_files.append(path)

    # Add important source files, scored by relevance to JD keywords
    source_candidates = []
    for path in file_paths:
        if (
            "test" in path.lower()
            or "doc" in path.lower()
            or path.split("/")[-1].lower() in manifest_files
        ):
            continue
        ext = "." + path.split(".")[-1] if "." in path else ""
        if ext in LANG_EXTENSIONS and ext not in {".yml", ".yaml", ".sh"}:
            source_candidates.append(path)

    scored_sources = [
        (path, _score_path_relevance(path, jd_keywords)) for path in source_candidates
    ]
    scored_sources.sort(key=lambda x: x[1], reverse=True)
    top_sources = [path for path, score in scored_sources[:3]]
    candidate_files.extend(top_sources)

    # Read contents (up to max_files_per_repo)
    # File content fetches are sequential within a repo because they share a budget
    repo_deps: set[str] = set()
    code_samples: list[str] = []

    for path in candidate_files[: settings.max_files_per_repo]:
        content = await client.get_file_content(repo.owner, repo.name, path)
        if not content:
            continue

        # Extract dependencies
        if path.split("/")[-1].lower() in manifest_files:
            repo_deps.update(_extract_dependencies_from_manifest(path, content))
        else:
            repo_deps.update(_parse_dependencies_from_code(path, content))

        # If it's a source code file and we have remaining token budget, add it to code samples
        is_readme = path.split("/")[-1].lower() in {"readme.md", "readme.txt", "readme"}
        is_source = not is_readme and path.split("/")[-1].lower() not in manifest_files

        remaining = remaining_char_budget_container[0]
        if is_source and remaining > 200:
            snippet_len = min(1500, remaining - 100)
            snippet = _extract_relevant_snippet(content, jd_keywords, snippet_len)
            formatted_snippet = f"File: {path}\n```\n{snippet}\n```"
            code_samples.append(formatted_snippet)
            remaining_char_budget_container[0] -= len(formatted_snippet)
        elif is_readme and remaining > 200:
            snippet = content[: min(1000, remaining - 100)]
            formatted_snippet = f"README Preview:\n{snippet}"
            code_samples.append(formatted_snippet)
            remaining_char_budget_container[0] -= len(formatted_snippet)

    dependency_summary = ", ".join(sorted(repo_deps)) if repo_deps else "None detected"
    if len(dependency_summary) > 200:
        dependency_summary = dependency_summary[:200] + "..."

    from agent.tools.repo_focus import infer_repo_type_tags_from_signals

    framework_markers: list[str] = []
    signal_blob = " ".join(file_paths).lower() + " " + " ".join(repo_deps).lower()
    for marker in (
        "fastapi",
        "flask",
        "django",
        "langchain",
        "streamlit",
        "react",
        "nextjs",
        "express",
    ):
        if marker in signal_blob:
            framework_markers.append(marker)

    base_tags: list[str] = []
    if project_type == "web-app":
        base_tags.append("frontend_app")
    elif project_type == "dockerized-app":
        base_tags.append("backend_service")
    elif project_type == "library":
        base_tags.append("sdk_or_library")
    elif code_file_count > 10:
        base_tags.append("automation_tool")
    else:
        base_tags.append("research_prototype")
    if has_docker and "backend_service" not in base_tags:
        base_tags.append("infra_iac_repo")

    architecture_layers: list[str] = []
    lower_paths = " ".join(file_paths).lower()
    if any(segment in lower_paths for segment in ("backend", "api", "services")):
        architecture_layers.append("services")
    if any(segment in lower_paths for segment in ("pipeline", "ingest", "chunk", "embed")):
        architecture_layers.append("pipeline")
    if any(segment in lower_paths for segment in ("data", "models", "schema")):
        architecture_layers.append("data")

    repo_type_tags = infer_repo_type_tags_from_signals(
        file_paths=file_paths,
        dependencies=repo_deps,
        framework_markers=framework_markers,
        architecture_layers=architecture_layers,
        base_tags=base_tags,
    )

    commits_dicts = [
        {"sha": c.sha, "message": c.message, "author_name": c.author_name, "date": c.date}
        for c in commits
    ]

    return RepoAnalysis(
        name=repo.name,
        url=repo.url,
        description=repo.description,
        languages=languages_pct,
        stars=repo.stars,
        is_fork=repo.is_fork,
        project_type=project_type,
        has_tests=has_tests,
        has_ci=has_ci,
        has_docs=has_docs,
        has_docker=has_docker,
        dependency_summary=dependency_summary,
        code_samples=code_samples,
        commit_frequency=commit_frequency,
        commit_quality=commit_quality,
        complexity_estimate=complexity,
        commits=commits_dicts,
        repo_type_tags=sorted(dict.fromkeys(repo_type_tags)),
        github_metadata=_build_repo_github_metadata(repo),
    )


async def analyze_github_repos(
    username: str,
    repo_urls: list[str] | None = None,
    discovered_repo_urls: list[str] | None = None,
    jd_structured: dict[str, Any] | None = None,
    *,
    sandbox_mode: Literal["inline", "deferred"] = "inline",
) -> dict[str, Any]:
    """Fetch and deeply analyze a candidate's GitHub repositories.

    Performance optimizations:
    - Uses a shared ``GitHubClient`` connection pool (single TCP/TLS session).
    - Parallelizes user events + user repos fetch with ``asyncio.gather``.
    - Parallelizes per-repo analysis (languages, tree, commits) with ``asyncio.gather``.
    - Parallelizes across repos with ``asyncio.gather``.

    Args:
        username: The candidate's GitHub username.
        repo_urls: A list of repository URLs parsed from their resume.
        jd_structured: Structured job description to guide selection and relevance.
    """
    settings = get_settings()
    resume_repo_urls = extract_github_repo_urls(repo_urls or [])
    discovered_repo_urls = extract_github_repo_urls(discovered_repo_urls or [])
    all_repo_urls = merge_github_repo_urls(resume_repo_urls, discovered_repo_urls)
    if not settings.github_analysis_enabled:
        logger.info("GitHub deep analysis is disabled by configuration.")
        return asdict(
            GitHubAnalysis(
                username=username,
                total_public_repos=0,
                total_stars=0,
                primary_languages=[],
                repo_analyses=[],
                coding_style_summary="GitHub analysis disabled by configuration.",
                overall_github_signal="none",
                collaboration_summary="GitHub analysis disabled by configuration.",
                commit_hygiene="GitHub analysis disabled by configuration.",
                resume_github_repo_urls=resume_repo_urls,
                discovered_github_repo_urls=discovered_repo_urls,
                selected_sandbox_repo_urls=[],
                sandbox_reports=[],
                repo_selection_mode="disabled",
            )
        )

    async with GitHubClient() as client:
        logger.info(f"Starting GitHub deep analysis for user: {username}")

        # --- Parallel fetch: user events + user repos + profile metadata ---
        events, all_repos, user_meta, profile_readme = await asyncio.gather(
            client.get_user_events(username),
            client.get_user_repos(username),
            client.get_user(username),
            client.get_profile_readme(username),
        )
        user_profile = _user_profile_to_dict(user_meta)
        collab_data = _analyze_collaboration(username, events)

        if not all_repos:
            logger.warning(f"No repositories found for GitHub user {username}")
            selected_sandbox_urls, selection_mode = _select_sandbox_repo_urls(
                ranked_repos=[],
                resume_repo_urls=all_repo_urls,
                settings=settings,
            )
            sandbox_reports = (
                []
                if sandbox_mode == "deferred"
                else await _evaluate_sandbox_repos(selected_sandbox_urls, settings)
            )
            activity_timeline = _build_activity_timeline(
                resume_repo_urls=all_repo_urls,
                repo_analyses=[],
                all_repos=[],
            )
            return asdict(
                GitHubAnalysis(
                    username=username,
                    total_public_repos=0,
                    total_stars=0,
                    primary_languages=[],
                    repo_analyses=[],
                    coding_style_summary=f"No repositories found for GitHub user {username}.",
                    overall_github_signal="none",
                    collaboration_summary=collab_data["summary"],
                    commit_hygiene="No commits to analyze.",
                    resume_github_repo_urls=resume_repo_urls,
                    discovered_github_repo_urls=discovered_repo_urls,
                    selected_sandbox_repo_urls=selected_sandbox_urls,
                    sandbox_reports=sandbox_reports,
                    repo_selection_mode=selection_mode,
                    user_profile=user_profile,
                    profile_readme=profile_readme or "",
                    activity_timeline=activity_timeline,
                )
            )

        # Rank all repos
        jd_keywords = _get_jd_keywords(jd_structured)
        repos_with_scores = [(repo, _score_repo_relevance(repo, jd_keywords)) for repo in all_repos]
        repos_with_scores.sort(key=lambda x: (x[1], x[0].stars, x[0].updated_at), reverse=True)

        ranked_repos = [repo for repo, score in repos_with_scores]
        resolved_resume_repos = await _resolve_resume_repos_for_analysis(
            client=client,
            all_repos=all_repos,
            resume_repo_urls=all_repo_urls,
            settings=settings,
        )
        selected_repos, selection_mode = _select_static_repos(
            resolved_resume_repos=resolved_resume_repos,
            ranked_repos=ranked_repos,
            resume_repo_urls=all_repo_urls,
            settings=settings,
        )
        selected_sandbox_urls, sandbox_selection_mode = _select_sandbox_repo_urls(
            ranked_repos=ranked_repos,
            resume_repo_urls=all_repo_urls,
            settings=settings,
        )

        # Calculate totals
        sum_stars = sum(r.stars for r in all_repos)

        # Aggregate languages
        lang_totals: dict[str, int] = {}
        for r in all_repos:
            if r.language:
                lang_totals[r.language] = lang_totals.get(r.language, 0) + 1
        sorted_langs = sorted(lang_totals.items(), key=lambda x: x[1], reverse=True)
        top_languages = [lang for lang, count in sorted_langs[:5]]

        # --- Parallel repo analysis ---
        # Shared mutable budget container (soft cap, minor race is acceptable)
        remaining_char_budget = [settings.github_content_token_cap]

        repo_analyses_list: list[RepoAnalysis] = list(
            await asyncio.gather(
                *[
                    _analyze_single_repo(repo, client, jd_keywords, settings, remaining_char_budget)
                    for repo in selected_repos
                ]
            )
        )

        repos_dict_list = [asdict(r) for r in repo_analyses_list]
        candidate_tags = _classify_candidate_tags(jd_structured, repos_dict_list)
        activity_timeline = _build_activity_timeline(
            resume_repo_urls=all_repo_urls,
            repo_analyses=repo_analyses_list,
            all_repos=all_repos,
        )
        github_metadata = {
            "total_public_repos": len(all_repos),
            "total_stars": sum_stars,
            "primary_languages": top_languages,
        }

        # Determine whether we should run the sandbox dynamically (auto/hybrid mode)
        run_sandbox = False
        enabled_val = getattr(settings, "github_clone_analysis_enabled", False)
        logger.info(
            "GITHUB_CLONE_ANALYSIS_ENABLED value: %s (type: %s)",
            enabled_val,
            type(enabled_val),
        )
        logger.info(f"resume_repo_urls: {resume_repo_urls}")
        logger.info(f"discovered_repo_urls: {discovered_repo_urls}")
        logger.info(f"selected_sandbox_urls: {selected_sandbox_urls}")
        if isinstance(enabled_val, bool):
            run_sandbox = enabled_val
        elif str(enabled_val).strip().lower() in ("true", "1", "yes", "on"):
            run_sandbox = True
        elif str(enabled_val).strip().lower() in ("false", "0", "no", "off"):
            run_sandbox = False
        elif str(enabled_val).lower() in ("auto", "hybrid"):
            # Dynamic hybrid rules
            if all_repo_urls:
                logger.info(
                    "Hybrid Sandbox: Enabled because candidate's resume "
                    "emphasizes specific projects."
                )
                run_sandbox = True
            elif jd_structured and any(
                word in str(jd_structured.get("job_title") or "").lower()
                for word in ["senior", "lead", "staff", "principal", "architect", "sr."]
            ):
                logger.info("Hybrid Sandbox: Enabled because screening is for a senior/lead role.")
                run_sandbox = True
            else:
                _, static_signal, _, _ = _generate_coding_style_summary_heuristic(
                    username, repos_dict_list, collab_data["summary"]
                )
                if static_signal == "moderate":
                    logger.info(
                        "Hybrid Sandbox: Enabled because static API signal is 'moderate' and "
                        "sandbox execution could elevate it."
                    )
                    run_sandbox = True
                else:
                    logger.info(
                        "Hybrid Sandbox: Disabled (static API signal is sufficient/no senior or "
                        "specific resume project cues)."
                    )

        if run_sandbox and selected_sandbox_urls and sandbox_mode == "inline":
            sandbox_task = asyncio.create_task(
                _evaluate_sandbox_repos(selected_sandbox_urls, settings)
            )
        else:
            sandbox_task = None

        if settings.github_llm_summary_enabled:
            summary_task = asyncio.create_task(
                _generate_coding_style_summary(
                    username, repos_dict_list, collab_data["summary"], settings
                )
            )
            if sandbox_task:
                sandbox_reports, summary_result = await asyncio.gather(sandbox_task, summary_task)
            else:
                sandbox_reports, summary_result = [], await summary_task
            coding_style_summary, overall_github_signal, collaboration_style, commit_hygiene = (
                summary_result
            )
        else:
            coding_style_summary, overall_github_signal, collaboration_style, commit_hygiene = (
                _generate_coding_style_summary_heuristic(
                    username, repos_dict_list, collab_data["summary"]
                )
            )
            if sandbox_task:
                sandbox_reports = await sandbox_task
            else:
                sandbox_reports = []

    return asdict(
        GitHubAnalysis(
            username=username,
            total_public_repos=len(all_repos),
            total_stars=sum_stars,
            primary_languages=top_languages,
            repo_analyses=repo_analyses_list,
            coding_style_summary=coding_style_summary,
            overall_github_signal=overall_github_signal,
            collaboration_summary=collaboration_style,
            commit_hygiene=commit_hygiene,
            resume_github_repo_urls=resume_repo_urls,
            discovered_github_repo_urls=discovered_repo_urls,
            selected_sandbox_repo_urls=selected_sandbox_urls if run_sandbox else [],
            sandbox_reports=sandbox_reports,
            repo_selection_mode=(sandbox_selection_mode if run_sandbox else selection_mode),
            candidate_tags=candidate_tags,
            github_metadata=github_metadata,
            user_profile=user_profile,
            profile_readme=profile_readme or "",
            activity_timeline=activity_timeline,
        )
    )
