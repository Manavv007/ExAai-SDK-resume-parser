# EXAai-ADK — Architecture (Google ADK + Exa)

## Production flow (`SCREENING_MODE=agent`, default)

```text
POST /screen (multipart: resume + JD)
    → prepare_screening_state (parse, PII redact, rubric, profile trust map)
    → ADK Runner + screening Agent (Gemini)
        → fetch_profiles (selected allowlisted URLs; Exa inside tool)
        → optional list_candidate_profile_urls only if meta needed
        → submit_screening_result
    → normalize + caps + validate_result → audit log → JSON response
```

The **judging model** decides **which allowlisted URLs** are worth fetching. It does not call Exa directly — it calls ADK **`FunctionTool`** wrappers that run SSRF, allowlist, cache, `exa-py`, and sanitization.

Set `SCREENING_MODE=pipeline` to use the legacy path: enrich **all** profile URLs, then one Gemini scoring call. Same prep, caps, and validation on both paths.

## Two layers

| Layer | Technology | Role |
|-------|------------|------|
| **Prep** | Python (`agent/prep.py`) | Always runs first: parse PDF/DOCX, `jd_structured`, redact resume, extract URLs, identity trust → session state |
| **Screening** | `google-adk` `Agent` + `Runner` (default) or pipeline scorer | Gemini orchestrates tools or scores enriched bundle |

## ADK agent tools

| Tool | Purpose |
|------|---------|
| `list_candidate_profile_urls` | Optional — URLs + trust already in brief; use for `profile_url_meta` only |
| `fetch_profiles` | Batch fetch allowlisted URLs (skips `scoring_untrusted`; session dedup + budget) |
| `submit_screening_result` | Writes `resume-screening-result-v1` into session; triggers normalize + caps |

Prep modules (`parser`, `link_extractor`, `pii_redactor`, `rubric_builder`, etc.) stay plain Python for unit tests; enrichment and scoring are invoked through tools or the pipeline path.

## Security inside tools (non-negotiable)

Every `fetch_profiles` URL:

1. SSRF guard (`security/ssrf_guard.py`)
2. Domain allowlist (`security/allowlist.py`)
3. Exa fetch or SQLite cache (`cache/url_cache.py`)
4. Sanitize + delimit content before returning to the model

Untrusted identity URLs (e.g. famous-name GitHub handles) are **not** fetched for scoring — stub metadata only.

## Config

```env
SCREENING_MODE=agent          # default; pipeline = legacy enrich-all path
MAX_URLS_PER_RESUME=10        # resume cap + agent session fetch budget
MAX_AGENT_TURNS=8
AGENT_RUN_TIMEOUT_SECONDS=120
```

Rollback: set `SCREENING_MODE=pipeline` in `.env` and restart — no code deploy.

## vs fully deterministic pipeline

| Approach | Pros | Cons |
|----------|------|------|
| **LLM + tools (default)** | Fewer Exa calls; model skips low-value URLs | Extra LLM turns; needs tool discipline |
| **Fixed enrich-all (pipeline)** | Predictable single scoring call | Higher Exa cost; fetches every link |

We combine both: **prep is fixed**; **enrichment is LLM-driven by default**, with pipeline as fallback.
