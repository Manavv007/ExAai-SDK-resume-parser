# EXAai-ADK — Implementation Progress

Track progress against [`implement.md`](./implement.md). Check boxes when a phase (or sub-item) is fully done and merged.

**Legend:** `- [ ]` not started · `- [x]` completed

---

## Phase 0 — Planning and contracts

- [x] Review [`json-schema.md`](./json-schema.md) with main platform team (fields, enums, API handoff) — documented in [`CONTRACTS.md`](./CONTRACTS.md)
- [x] Confirm LLM provider and model ID (Gemini API + `GEMINI_MODEL_ID`)
- [x] Confirm Exa API access and quotas — documented; production keys pending team
- [x] Align `recommendation` enum (`advance` | `hold` | `reject`) with hiring workflow

---

## Phase 1 — Repository bootstrap

- [x] Create directory structure per `implement.md` Step 1
- [x] Add `pyproject.toml` with pinned dependencies
- [x] Add `.env.example` and document required env vars
- [x] Add `.gitignore` (`.env`, `__pycache__`, `.venv`, cache DB)
- [x] README: how to run locally and call `/screen`

---

## Phase 2 — Output schema (block other phases)

- [x] Create `agent/schema/resume-screening-result-v1.json` from [`json-schema.md`](./json-schema.md)
- [x] Add Pydantic models mirroring the schema
- [x] Implement `validate_result()` in `agent/tools/validator.py`
- [x] Unit tests: valid fixture passes, invalid fixtures fail
- [x] Wire schema validation into CI (pytest on push)

---

## Phase 3 — Security layer

- [x] `security/pii_redactor.py` — Presidio entities and placeholders
- [x] Unit tests: all target entity types redacted; summary counts correct
- [x] `security/ssrf_guard.py` — HTTPS, DNS, private IP rejection
- [x] Unit tests: localhost, RFC1918, raw IP hostname blocked
- [x] `security/allowlist.py` — categorized domain allowlist
- [x] Unit tests: allowed vs blocked domains

---

## Phase 4 — Parsing and link extraction

- [x] `tools/parser.py` — PDF, DOCX, plain text
- [x] PDF hyperlink extraction
- [x] JD structured extraction → `jd_structured`
- [x] `tools/link_extractor.py` — explicit URLs + handle patterns
- [x] Platform inference by domain
- [x] Link dedupe and `MAX_URLS_PER_RESUME` cap
- [x] Unit tests for parser and link extractor fixtures

---

## Phase 5 — Crawl, cache, and sanitize

- [x] `cache/url_cache.py` — TTL cache (SQLite)
- [x] `tools/crawler.py` — Exa client (used by enrichment)
- [x] Cache hit/miss behavior and TTL expiry tests
- [x] `tools/sanitizer.py` — HTML strip, injection regex, delimiters
- [x] Unit tests: injection patterns stripped; delimiters present

---

## Phase 6 — Rubric and scoring

- [x] `tools/rubric_builder.py` — must-have / nice-to-have from `jd_structured`
- [x] Bias-avoidance preamble in session state
- [x] `tools/scorer.py` — Gemini JSON output → platform schema shape
- [x] Map LLM output to `requirement_matches` and `resume_similarity_score`
- [x] `recommendation` + `recommendation_reasoning` populated
- [x] Scorer retry on malformed JSON (one in-call fix)
- [x] Unit tests: rubric builder; mocked scorer output validates

---

## Phase 7 — ADK pipeline and validation

- [x] `agent/pipeline.py` — prep → enrich → score → validate → audit
- [x] Session state contract documented in `agent/session_state.py`
- [x] Validation retry once (`score_with_validation`); second failure → `failed` + `errors[]`
- [x] `audit/logger.py` — structured JSON log, no PII
- [x] Integration test: full pipeline with mocked Exa + Gemini

---

## Phase 8 — HTTP API

- [x] `api/main.py` + `routes.py` — `POST /screen`
- [x] Multipart upload, 5MB limit, magic-byte file type check
- [x] Require `application_id` and `job_id` in request
- [x] `api/middleware.py` — Bearer auth, `X-Request-ID`, `X-Processing-Time-Ms`
- [x] `GET /health`
- [ ] Manual smoke test: local `uvicorn` + sample resume/JD

---

## Phase 9 — Test suite and hardening

- [x] Integration test: software engineer fixture
- [x] Integration test: graphic designer fixture
- [x] Integration test: academic researcher fixture
- [x] Security test: SSRF private IP blocked
- [x] Security test: crawl injection does not inflate score or leak into evidence
- [x] Assert no PII from fixtures appears in API response
- [x] Optional: `Dockerfile` + non-root runtime smoke test (`RUN_DOCKER_TESTS=1` for build)

---

## Phase 12 — ADK agent migration (`docs/AGENT_MIGRATION.md`)

- [x] Phase 1: `fetch_profiles` batch tool
- [x] Phase 2: `submit_screening_result` tool
- [x] Phase 3: Agent definition + instruction + user message builder
- [x] Phase 4: `run_screening_agent_async` + Runner integration
- [x] Phase 5: `SCREENING_MODE` feature flag + API branch
- [x] Phase 6: Agent-path integration tests (domain + security), docs, default `agent`

---

## Phase 10 — Main platform integration

- [ ] Document request/response for main app team
- [ ] Main app: call agent `POST /screen` (or embed library)
- [ ] Main app: map response to `POST /api/applications/update-score`
- [ ] Main app: set `resume_screening_status` (`processing` → `completed` | `failed`)
- [ ] Optional: extend `ResumeSimilarityScore` TypeScript type on main repo
- [ ] End-to-end test on staging with real `application_id` / `job_id`

---

## Phase 11 — Release

- [ ] Version tag `AGENT_VERSION` in metadata
- [ ] Production env vars documented (no secrets in repo)
- [ ] Runbook: failures, Exa timeouts, LLM rate limits
- [ ] Sign-off from main platform that jsonb payload matches expectations

---

## Quick status

| Phase | Name | Status |
|-------|------|--------|
| 0 | Planning and contracts | Done |
| 1 | Repository bootstrap | Done |
| 2 | Output schema | Done |
| 3 | Security layer | Done |
| 4 | Parsing and links | Done |
| 5 | Crawl, cache, sanitize | Done |
| 6 | Rubric and scoring | Done |
| 7 | ADK pipeline | Done |
| 8 | HTTP API | Done |
| 9 | Tests and hardening | Done |
| 10 | Main platform integration | Deferred |
| 11 | Release | Not started |

_Update the table Status column as phases complete (e.g. In progress / Done)._
