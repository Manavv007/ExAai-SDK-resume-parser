"""Deterministic JD and resume structuring (no LLM)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent.tools.parser import JdRequirement, JdStructured

_BULLET = re.compile(r"^[\-*•]\s+(.+)$")
_NUMBERED = re.compile(r"^\d+[\.\)]\s+(.+)$")
_INLINE_MUST = re.compile(
    r"(?i)^(?:must\s+have|required|essential|minimum)\s*:\s*(.+)$"
)
_INLINE_NICE = re.compile(
    r"(?i)^(?:preferred|nice\s+to\s+have|nice-to-have|bonus|plus)\s*:\s*(.+)$"
)
_YEARS = re.compile(r"(?i)(\d+)\+?\s*(?:years?|yrs?)\b")
_EMAIL = re.compile(r"@")
_URL = re.compile(r"https?://|www\.|\.com/|\.io/", re.I)
_TITLE_HINT = re.compile(
    r"(?i)\b(engineer|developer|designer|manager|analyst|researcher|architect|"
    r"scientist|consultant|lead|director|nurse|writer|specialist)\b"
)

_SECTION_MUST = (
    "requirements",
    "qualifications",
    "must have",
    "must-have",
    "minimum qualifications",
    "what you need",
    "what we're looking for",
    "what we are looking for",
)
_SECTION_NICE = (
    "nice to have",
    "nice-to-have",
    "preferred",
    "preferred qualifications",
    "bonus",
    "pluses",
    "desired",
)
_SECTION_SKIP = (
    "about the role",
    "about us",
    "about",
    "company",
    "location",
    "benefits",
    "equal opportunity",
    "how to apply",
    "key responsibilities",
    "responsibilities",
    "what you'll do",
    "what you will do",
    "role overview",
)


@dataclass
class ResumeStructured:
    candidate_name: str | None = None
    headline: str | None = None
    skills: list[str] = field(default_factory=list)
    experience_years: int | None = None
    experience_highlights: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)


def normalize_extracted_text(text: str) -> str:
    """Reflow PDF/DOCX line breaks and collapse excessive blank lines."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    merged: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if merged and merged[-1] != "":
                merged.append("")
            continue
        if (
            merged
            and merged[-1]
            and not merged[-1].endswith((".", ":", ";", "?", "!"))
            and stripped[0].islower()
        ):
            merged[-1] = f"{merged[-1]} {stripped}"
        else:
            merged.append(stripped)
    return "\n".join(merged).strip()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _normalize_header(line: str) -> str:
    return line.strip().rstrip(":").strip().lower()


def _section_kind(header: str) -> str | None:
    h = _normalize_header(header)
    if not h:
        return None
    if any(h == s or h.startswith(s) for s in _SECTION_MUST):
        return "must"
    if any(h == s or h.startswith(s) for s in _SECTION_NICE):
        return "nice"
    if any(h == s or h.startswith(s) for s in _SECTION_SKIP):
        return "skip"
    return None


def _strip_bullet(line: str) -> str:
    numbered = _NUMBERED.match(line)
    if numbered:
        return numbered.group(1).strip()
    bullet = _BULLET.match(line)
    if bullet:
        return bullet.group(1).strip()
    return line.strip()


def _is_noise_line(line: str) -> bool:
    if len(line) > 220:
        return True
    lowered = line.lower()
    if lowered.startswith(("we are ", "you will ", "our team", "the ideal")):
        return True
    if _EMAIL.search(line) and "@" in line.split()[0] if line.split() else False:
        return False
    return False


def _split_skill_phrases(text: str) -> list[str]:
    """Split short comma/semicolon-separated skill lists."""
    if len(text) > 120 or text.count(",") + text.count(";") < 1:
        return [text]
    parts = re.split(r"[,;]\s*", text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]


def _classify_requirement_type(text: str, domain: str) -> str:
    from agent.tools.rubric_builder import infer_requirement_type

    return infer_requirement_type(text, domain)


def _requirements_with_types(
    must: list[str],
    nice: list[str],
    domain: str,
) -> list[JdRequirement]:
    items: list[JdRequirement] = []
    for text in must:
        stripped = text.strip()
        if not stripped:
            continue
        items.append(
            JdRequirement(
                text=stripped,
                weight="must_have",
                requirement_type=_classify_requirement_type(stripped, domain),
            )
        )
    for text in nice:
        stripped = text.strip()
        if not stripped:
            continue
        items.append(
            JdRequirement(
                text=stripped,
                weight="nice_to_have",
                requirement_type=_classify_requirement_type(stripped, domain),
            )
        )
    return items


def extract_jd_requirements(jd_text: str) -> tuple[list[str], list[str]]:
    """Parse must-have and nice-to-have requirement strings from JD text."""
    must: list[str] = []
    nice: list[str] = []
    section: str | None = None

    for raw_line in jd_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        header_kind = _section_kind(line)
        if header_kind:
            section = header_kind
            continue

        if _is_noise_line(line):
            continue

        content = _strip_bullet(line)
        if not content:
            continue

        inline_must = _INLINE_MUST.match(content)
        if inline_must:
            must.extend(_split_skill_phrases(inline_must.group(1).strip()))
            continue

        inline_nice = _INLINE_NICE.match(content)
        if inline_nice:
            nice.extend(_split_skill_phrases(inline_nice.group(1).strip()))
            continue

        if section == "must":
            must.extend(_split_skill_phrases(content))
        elif section == "nice":
            nice.extend(_split_skill_phrases(content))
        elif section != "skip":
            lowered = content.lower()
            if any(k in lowered for k in ("must have", "required", "essential")):
                must.extend(_split_skill_phrases(content))
            elif any(k in lowered for k in ("preferred", "nice to have", "bonus")):
                nice.extend(_split_skill_phrases(content))

    return _dedupe(must), _dedupe(nice)


def detect_jd_domain(text: str) -> str:
    keywords: dict[str, tuple[str, ...]] = {
        "technical": (
            "software",
            "engineer",
            "developer",
            "python",
            "backend",
            "frontend",
            "devops",
            "machine learning",
            "data scientist",
            "rag",
            "api",
        ),
        "design": (
            "designer",
            "ux",
            "ui",
            "graphic",
            "figma",
            "creative",
            "visual",
        ),
        "academic": (
            "research",
            "professor",
            "phd",
            "postdoc",
            "university",
            "publication",
        ),
        "writing": ("writer", "content", "editor", "copywriter", "journalist"),
        "business": (
            "product manager",
            "consultant",
            "startup",
            "founder",
            "sales",
        ),
        "healthcare": ("nurse", "rn ", "clinical", "patient", "hospital", "healthcare"),
    }
    lowered = text.lower()
    scores = {
        domain: sum(1 for kw in kws if kw in lowered)
        for domain, kws in keywords.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def detect_jd_seniority(text: str) -> str | None:
    lowered = text.lower()
    if re.search(r"\bmid[\s-]?level\b", lowered):
        return "mid"
    if re.search(r"\bentry[\s-]?level\b", lowered):
        return "entry"
    for level in ("principal", "staff", "senior", "lead", "junior", "intern"):
        if re.search(rf"\b{re.escape(level)}\b", lowered):
            return level
    return None


def detect_jd_industry(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("company:"):
            return stripped.split(":", 1)[1].strip() or None
        if stripped.lower().startswith("industry:"):
            return stripped.split(":", 1)[1].strip() or None
    return None


def extract_jd_title(jd_text: str) -> str | None:
    for line in jd_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) > 120:
            continue
        if stripped.startswith(("-", "*", "•")):
            continue
        lowered = stripped.lower()
        if any(
            lowered.startswith(prefix)
            for prefix in (
                "company:",
                "location:",
                "experience:",
                "about",
                "requirements",
                "nice to have",
            )
        ):
            continue
        if _section_kind(stripped):
            continue
        return stripped
    return None


def parse_jd_local(jd_text: str) -> JdStructured:
    """Full local JD structuring with typed requirements."""
    normalized = normalize_extracted_text(jd_text)
    must, nice = extract_jd_requirements(normalized)
    domain = detect_jd_domain(normalized)
    return JdStructured(
        job_title=extract_jd_title(normalized),
        domain=domain,
        industry=detect_jd_industry(normalized),
        seniority=detect_jd_seniority(normalized),
        must_have=must,
        nice_to_have=nice,
        requirements=_requirements_with_types(must, nice, domain),
    )


def _looks_like_name(line: str) -> bool:
    if _EMAIL.search(line) or _URL.search(line):
        return False
    if len(line) > 60 or len(line.split()) > 5:
        return False
    words = line.split()
    if len(words) < 2:
        return False
    alpha = sum(ch.isalpha() for ch in line)
    return alpha / max(len(line), 1) > 0.6


def _looks_like_headline(line: str) -> bool:
    if len(line) > 100 or _EMAIL.search(line):
        return False
    return bool(_TITLE_HINT.search(line))


def _parse_skills_section(lines: list[str]) -> list[str]:
    skills: list[str] = []
    in_section = False
    for line in lines:
        header = _normalize_header(line)
        if header in ("skills", "technical skills", "technologies", "tools", "core skills"):
            in_section = True
            continue
        if in_section:
            if _section_kind(line) or (header.endswith(":") and header not in ("skills",)):
                break
            content = _strip_bullet(line)
            if content:
                skills.extend(_split_skill_phrases(content))
    return _dedupe(skills)


def _parse_education_section(lines: list[str]) -> list[str]:
    education: list[str] = []
    in_section = False
    for line in lines:
        header = _normalize_header(line)
        if header in ("education", "academic background", "degrees"):
            in_section = True
            continue
        if in_section:
            if _section_kind(line) or (":" in line and not _strip_bullet(line)):
                if header not in ("education", "academic background", "degrees"):
                    break
            content = _strip_bullet(line)
            if content and len(content) < 200:
                education.append(content)
    return _dedupe(education)


def _parse_experience_highlights(lines: list[str]) -> list[str]:
    highlights: list[str] = []
    in_section = False
    for line in lines:
        header = _normalize_header(line)
        if header in (
            "experience",
            "work experience",
            "professional experience",
            "employment",
        ):
            in_section = True
            continue
        if in_section:
            if header in ("education", "skills", "projects", "certifications"):
                break
            content = _strip_bullet(line)
            if content and len(content) < 300:
                highlights.append(content)
    return _dedupe(highlights[:8])


def _infer_skills_from_text(text: str) -> list[str]:
    """Pull likely skills from experience summary lines."""
    skill_tokens: list[str] = []
    tech_pattern = re.compile(
        r"\b(Python|Java(?:Script)?|TypeScript|Go|Rust|SQL|PostgreSQL|Postgres|"
        r"FastAPI|Flask|Django|React|Vue|Angular|Kubernetes|Docker|AWS|GCP|Azure|"
        r"Redis|MongoDB|GraphQL|gRPC|LangChain|LlamaIndex|RAG|Figma|UX|UI)\b",
        re.I,
    )
    for match in tech_pattern.finditer(text):
        token = match.group(0)
        if token.lower() not in {s.lower() for s in skill_tokens}:
            skill_tokens.append(token if token.isupper() else token.title())
    return skill_tokens


def parse_resume_local(text: str) -> ResumeStructured:
    """Extract resume highlights locally (no LLM)."""
    normalized = normalize_extracted_text(text)
    lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]

    candidate_name: str | None = None
    headline: str | None = None
    for idx, line in enumerate(lines[:6]):
        if candidate_name is None and _looks_like_name(line):
            candidate_name = line
            continue
        if headline is None and _looks_like_headline(line):
            headline = line
            continue
        if headline is None and idx == 1 and len(line) < 100 and not _EMAIL.search(line):
            headline = line

    experience_highlights = _parse_experience_highlights(lines)
    skills = _parse_skills_section(lines)
    education = _parse_education_section(lines)

    years: int | None = None
    for line in lines:
        match = _YEARS.search(line)
        if match:
            years = max(years or 0, int(match.group(1)))

    if not skills:
        skills = _infer_skills_from_text(normalized)

    if not experience_highlights:
        for line in lines:
            if "experience" in line.lower() and ":" in line:
                tail = line.split(":", 1)[1].strip()
                if tail:
                    experience_highlights.append(tail)
                    break

    return ResumeStructured(
        candidate_name=candidate_name,
        headline=headline,
        skills=_dedupe(skills)[:25],
        experience_years=years,
        experience_highlights=experience_highlights,
        education=education,
    )
