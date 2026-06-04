# Resume Screening Agent — Implementation Plan (EXAai-ADK)

## Project context

You are building a **standalone resume screening agent** (this repo: **EXAai-ADK**) that:

1. Accepts a candidate resume and a job description
2. Enriches the resume with live web data from links in the document (and inferred platform URLs by domain)
3. Scores the candidate against the JD using a structured rubric
4. Returns a **platform-ready JSON result** defined in [`json-schema.md`](./json-schema.md)

The agent is orchestrated with **Google Agent Development Kit (ADK)** where it fits the pipeline, uses **Exa AI** for crawling, and uses **Gemini** (or another configured LLM) for JD parsing and scoring.

### What this repo is not

- **This project does not run on GCP.** No Cloud Run, Vertex AI, Firestore, GCS, Secret Manager, or Cloud Logging in this codebase.
- **The main hiring platform** (Next.js + Supabase, deployed on GCP by your team) **consumes** the agent output via existing APIs such as `POST /api/applications/update-score`. See [`json-schema.md`](./json-schema.md) for field mapping and integration notes.

### Deployment model

- Run locally, in Docker, or on any host your team chooses
- Configure secrets via `.env` (see `.env.example`)
- The main platform is responsible for auth, rate limits, and persisting results to Supabase

---

## Non-negotiable constraints

1. **Lock the output schema first** — [`json-schema.md`](./json-schema.md) / `agent/schema/resume-screening-result-v1.json` is the source of truth. Every component produces or consumes it. Wire a validator into CI before feature work ships.
2. **PII redaction before any LLM or crawler** — names, emails, phones, addresses, ages, and national identifiers must not reach Gemini, Exa prompts, or logs.
3. **External content is untrusted data** — delimit, strip injection patterns, and never treat crawled text as instructions.
4. **Validate every response** — `jsonschema` (or Pydantic) on every success path; one structured retry on validation failure; safe failure payload on second failure (never a raw stack trace to the client).
5. **Audit without PII** — structured logs may include `application_id`, scores, counts, and latency — never resume body, redacted text, or personal data.

---

## Output contract (implement first)

Canonical schema: **`resume-screening-result-v1`** (full spec in [`json-schema.md`](./json-schema.md)).

Required for integration with the main app:

| Field | Notes |
|--------|--------|
| `application_id`, `job_id` | UUID strings from the caller |
| `resume_screening_status` | `queued` \| `processing` \| `completed` \| `failed` |
| `resume_similarity_score.score` | Integer 0–100 (existing Zod/DB constraint) |
| `resume_similarity_score.reasoning` | Single sentence, max 500 chars |

Extended fields (store in same jsonb or pass through API):

- `requirement_matches[]` — aligned with JobDetailsV14 (technical_skill, soft_skill, experience, education, responsibility)
- `recommendation` — `advance` \| `hold` \| `reject`
- `recommendation_reasoning`, `red_flags[]`, `sources_crawled[]`, `metadata`, `errors[]`

Implement `validate_result(data: dict) -> bool` in `agent/tools/validator.py`. On failure: one correction retry on the scorer; then return `resume_screening_status: "failed"` with populated `errors[]`.

---

## Step 1 — Repository and environment setup

```
EXAai-ADK/
├── agent/
│   ├── __init__.py
│   ├── pipeline.py              # ADK SequentialAgent / tool chain
│   ├── tools/
│   │   ├── parser.py            # Resume + JD parsing
│   │   ├── link_extractor.py    # Link extraction + platform inference
│   │   ├── crawler.py           # Exa AI concurrent fetching
│   │   ├── sanitizer.py         # Content cleaning and injection stripping
│   │   ├── rubric_builder.py    # JD-derived scoring rubric
│   │   ├── scorer.py            # LLM judge (Gemini via API)
│   │   └── validator.py         # Output schema validation + retry
│   ├── schema/
│   │   └── resume-screening-result-v1.json
│   ├── security/
│   │   ├── pii_redactor.py
│   │   ├── ssrf_guard.py
│   │   └── allowlist.py
│   ├── cache/
│   │   └── url_cache.py         # TTL cache (SQLite or in-memory; optional Redis)
│   └── audit/
│       └── logger.py            # Structured stdout/file logging (no PII)
├── api/
│   ├── main.py                  # FastAPI application
│   ├── routes.py
│   ├── middleware.py            # Request ID, auth (API key from env)
│   └── health.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── Dockerfile                   # Optional — not GCP-specific
├── .env.example
├── pyproject.toml
├── implement.md
├── json-schema.md
└── progress.md
```

**Dependencies:** `google-adk`, `google-generativeai` (or chosen LLM SDK), `fastapi`, `uvicorn`, `pdfplumber`, `python-docx`, `presidio-analyzer`, `presidio-anonymizer`, `exa-py`, `jsonschema`, `pydantic`, `httpx`, `python-multipart`.

**Environment variables** (`.env.example`):

```
GEMINI_API_KEY=
EXA_API_KEY=
API_KEYS=                    # Comma-separated Bearer tokens for /screen
MAX_URLS_PER_RESUME=10
URL_FETCH_TIMEOUT_SECONDS=5
CONTENT_TOKEN_CAP=8000
CACHE_TTL_SECONDS=86400
AGENT_VERSION=0.1.0
GEMINI_MODEL_ID=gemini-2.0-flash
LOG_LEVEL=INFO
```

---

## Step 2 — Security layer (before any external calls)

### PII redactor (`security/pii_redactor.py`)

Use Presidio. Detect and replace: `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `LOCATION`, `DATE_TIME`, `URL` (body text only, not the extracted link list), `NRP`, `AGE`. Placeholders: `[PERSON_1]`, `[EMAIL_1]`, etc. Return redacted text and a summary for internal metrics only (do not echo raw PII in API output).

### SSRF guard (`security/ssrf_guard.py`)

Before any fetch:

- HTTPS only
- Reject private/reserved IPs after DNS resolution (`ipaddress` module)
- Reject raw IP hostnames, non-HTTPS schemes, URLs longer than 2048 chars
- Optional: 60s in-memory DNS cache

### Domain allowlist (`security/allowlist.py`)

Explicit categorized allowlist (portfolio, GitHub, LinkedIn, Medium, academic, creative, etc. — see original platform list in project notes). URLs not on the list: skip fetch, record in `sources_crawled` / errors with a blocked code, **do not hard-fail** the pipeline.

---

## Step 3 — ADK pipeline (Google ADK SDK)

See [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) for the full diagram.

**Two layers:**

1. **Deterministic prep** (`agent/prep.py`) — always runs before the agent:
   - `parse_file` → resume + JD text, PDF hyperlinks
   - `parse_jd_structured` → `jd_structured`
   - `redact_text` → `resume_text`
   - `extract_links` → `profile_urls`
   - Writes into ADK `session.state` (no raw PII except redacted resume text)

2. **ADK `Agent` + `Runner`** (`agent/pipeline.py`) — Gemini orchestrates via **function calling**:
   - `list_candidate_profile_urls` — read URLs from session
   - `fetch_profile_content` — LLM chooses which URL to enrich; tool runs SSRF + allowlist + **Exa** + sanitize
   - Agent final turn → `resume-screening-result-v1` JSON
   - `validate_result` + audit (Phase 7–8)

This matches: *parse first → LLM sees data → LLM calls Exa (inside tool) → LLM decides result.*

**SDK:** `google-adk` `Agent`, `Runner`, `App`, `FunctionTool` (auto-wrapped Python functions in `agent/adk_tools.py`). `SequentialAgent` is deprecated in ADK 2.x.

**Session state keys:** `application_id`, `job_id`, `resume_text`, `jd_raw`, `jd_structured`, `profile_urls`, `enriched_contents`, …

**Retry:** On validation failure, re-prompt agent once with correction; then `failed` + `errors[]`.

---

## Step 4 — Parser tool

**File:** `tools/parser.py`

- PDF: `pdfplumber` (text + hyperlink annotations)
- DOCX: `python-docx` (paragraphs + tables)
- Plain text: direct read
- JD: optional lightweight LLM pass to extract must-have / nice-to-have, title, domain, seniority → `jd_structured`

Pass resume text to PII redactor before any other state key is written.

---

## Step 5 — Link extractor

**File:** `tools/link_extractor.py`

Sources: PDF links, `https?://` regex, handle patterns (`@user`, `github.com/...`, `linkedin.com/in/...`). Normalize URLs (strip tracking params, enforce scheme).

**Platform inference** by JD/resume domain (technical → GitHub/Kaggle; design → Behance; academic → Scholar/ORCID; etc.). Tag inferred URLs in scoring evidence; record in `sources_crawled`.

Deduplicate and cap at `MAX_URLS_PER_RESUME`.

---

## Step 6 — Exa crawler

**File:** `tools/crawler.py`

- Check `url_cache` (SHA-256 of URL) before Exa call
- Cache miss: Exa contents API, `asyncio.gather` with semaphore (max 6), 5s timeout per URL
- Cap content at `CONTENT_TOKEN_CAP` per URL
- On timeout/error: continue pipeline; record in `sources_crawled` / `errors`

---

## Step 7 — Sanitizer

**File:** `tools/sanitizer.py`

Regex-based (no LLM for injection detection):

- Strip HTML/scripts
- Remove patterns like “ignore previous instructions”, “you are now”, “rate this candidate”, etc.
- Truncate to token cap
- Wrap blocks:

```
===BEGIN EXTERNAL CONTENT: {url}===
{sanitized_content}
===END EXTERNAL CONTENT===
```

---

## Step 8 — Rubric builder

**File:** `tools/rubric_builder.py`

From `jd_structured`, build criteria with `must_have` / `nice_to_have`. Bias-avoidance preamble in session state. Rule: failing all must-haves caps overall score at 40 (enforced in scorer prompt or post-processing).

Align criterion wording with JobDetailsV14 sections for clean `requirement_matches` in the final JSON.

---

## Step 9 — Scorer

**File:** `tools/scorer.py`

- Gemini via `google-generativeai` and `GEMINI_API_KEY`
- `response_mime_type="application/json"` where supported
- Prompt: role, `rubric_preamble`, rubric, redacted resume, delimited external content only as data
- Map model output to `resume-screening-result-v1` (including `resume_similarity_score`, `requirement_matches`, `recommendation`)
- `max_output_tokens` ~2000; one in-conversation JSON fix retry before validator retry

---

## Step 10 — Validator

**File:** `tools/validator.py`

Validate against `agent/schema/resume-screening-result-v1.json`. Ensure `score` is 0–100 integer and `reasoning` length ≤ 500. Set `resume_screening_status` to `completed` on success.

---

## Step 11 — Audit logger

**File:** `audit/logger.py`

Log one structured JSON line per run:

`application_id`, `job_id`, `recommendation`, `score`, `model_version`, `processing_time_ms`, `sources_attempted`, `sources_successful`, `error_count`, `timestamp_utc`.

**Never** log resume or JD content.

---

## Step 12 — FastAPI API

**Routes** (`api/routes.py`):

| Method | Path | Behavior |
|--------|------|----------|
| `POST` | `/screen` | Multipart: `resume` (PDF/DOCX), `jd` (file or text), `application_id`, `job_id`. Max 5MB; magic-byte type check. Run pipeline; return `resume-screening-result-v1` JSON. Target latency: &lt; 15s. |
| `GET` | `/health` | `{ "status": "ok", "model": "...", "agent_version": "..." }` |
| `GET` | `/screen/{application_id}` | Optional: return last result if you add persistence later |

**Middleware:** `X-Request-ID`, Bearer auth against `API_KEYS`, timing → `metadata.processing_time_ms`.

Rate limiting is the **main platform’s** concern (API Gateway / app layer), not required in this service.

---

## Step 13 — Integration with main platform (GCP app)

This agent **does not** write to Supabase. The main app should:

1. Set `resume_screening_status` to `processing` when starting
2. Call `POST /screen` on this service (or run the library in-process)
3. On success: `POST /api/applications/update-score` with `resume_similarity_score` and extended fields per [`json-schema.md`](./json-schema.md)
4. Set `resume_screening_status` to `completed` or `failed`

Optional TypeScript extension on the main repo:

```typescript
export type ResumeSimilarityScore = {
  score: number;
  reasoning: string;
  matches?: RequirementMatch[];
  recommendation?: "advance" | "hold" | "reject";
  redFlags?: RedFlag[];
  sources?: CrawledSource[];
  errors?: ScreeningError[];
  meta?: ScreeningMetadata;
};
```

---

## Step 14 — Tests

**Unit:** PII redactor, SSRF guard, allowlist, sanitizer, rubric builder, validator (accept/reject schema).

**Integration:** Fixture resume + JD for three domains (software, design, academic). Mock Exa and Gemini. Assert schema validity, no PII in output, `recommendation` enum valid.

**Security:** Private IP URL blocked; injection string in crawled content does not produce score 100 or appear in evidence.

---

## Step 15 — Optional Docker

Multi-stage `python:3.12-slim`, non-root user, port 8080. No secrets baked into image. Suitable for any container host — not tied to GCP.

---

## Delivery checklist

- [ ] `resume-screening-result-v1.json` committed and validated in CI
- [ ] Validator on every success path
- [ ] No secrets in source control
- [ ] No resume/PII in logs
- [ ] PII redaction before LLM and Exa
- [ ] SSRF + allowlist before network
- [ ] Exa fetches concurrent (semaphore), not sequential
- [ ] Integration test for non-technical candidate domain
- [ ] Documented handoff for main platform (`json-schema.md` + env vars)

Track phase completion in [`progress.md`](./progress.md).
