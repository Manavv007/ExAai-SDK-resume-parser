# EXAai-ADK — Platform contracts (Phase 0)

Locked decisions for this repo and handoff to the main hiring platform. Update this file when the platform team changes requirements.

## Scope

| Responsibility | EXAai-ADK (this repo) | Main platform (GCP / Next.js + Supabase) |
|----------------|------------------------|------------------------------------------|
| Resume + JD intake | `POST /screen` (multipart) | Upload storage, `application_id` / `job_id` |
| Crawl + score + PII | Yes | No |
| Persist score | Returns JSON only | `updateApplicationResumeScore`, `resume_screening_status` |
| Auth for screening API | Bearer `API_KEYS` | App auth for recruiters/candidates |
| Rate limits | Optional at reverse proxy | Primary |

Full field spec: [`json-schema.md`](./json-schema.md).

## Output schema

- **ID:** `resume-screening-result-v1`
- **Machine-readable:** [`agent/schema/resume-screening-result-v1.json`](./agent/schema/resume-screening-result-v1.json)
- **Pydantic models:** `agent/schema/models.py` · `validate_result()` in `agent/tools/validator.py`
- **Required for DB compatibility:**
  - `application_id`, `job_id` (UUID strings from caller)
  - `resume_screening_status`: `queued` | `processing` | `completed` | `failed`
  - `resume_similarity_score.score`: integer 0–100
  - `resume_similarity_score.reasoning`: string, max 500 characters

## Recommendation enum (aligned)

| Value | Meaning | Suggested platform action |
|-------|---------|---------------------------|
| `advance` | Strong fit | Move toward interview / next stage |
| `hold` | Unclear or partial fit | Manual review |
| `reject` | Poor fit or hard fails | Reject or archive prospect |

Maps to extended `ResumeSimilarityScore.recommendation` on the main app; does **not** replace `job_applications.status` (`accepted` / `rejected` workflow).

## LLM provider (confirmed for this repo)

| Setting | Value |
|---------|--------|
| Provider | Google Gemini via **Gemini API** (not Vertex AI) |
| SDK | `google-generativeai` |
| Env | `GEMINI_API_KEY`, `GEMINI_MODEL_ID` |
| Default model | `gemini-2.0-flash` |
| Uses | JD structuring (light pass), final scoring JSON |

**Why not Vertex:** This service is not deployed on GCP. Vertex remains an option only if this repo is later hosted on GCP with enterprise policy requirements.

## Exa AI (confirmed)

| Setting | Value |
|---------|--------|
| SDK | `exa-py` |
| Env | `EXA_API_KEY` |
| Use | Fetch public profile/portfolio URLs after allowlist + SSRF checks |
| Limits | `MAX_URLS_PER_RESUME`, `URL_FETCH_TIMEOUT_SECONDS`, `CONTENT_TOKEN_CAP` |

Obtain API key: https://dashboard.exa.ai — confirm quota/billing with your team before production.

## Main platform integration flow

```text
1. Main app sets resume_screening_status = "processing"
2. Main app POST /screen → EXAai-ADK
   Body: application_id, job_id, resume file, jd file|text
   Header: Authorization: Bearer <key from API_KEYS>
3. On 200 + resume_screening_status "completed":
   POST /api/applications/update-score
   { application_id, resume_similarity_score: { score, reasoning, ...extended } }
   Set resume_screening_status = "completed"
4. On failure:
   Set resume_screening_status = "failed"
   Store errors[] from agent response if present
```

## Open items for platform team sign-off

- [ ] Confirm extended jsonb fields (`requirement_matches`, `red_flags`, `sources_crawled`) are stored inside `resume_similarity_score` vs new column
- [ ] Staging URL for EXAai-ADK service and shared `API_KEYS` rotation process
- [ ] Whether `hold` triggers a specific UI state or label

_Last updated: Phase 0 implementation._
