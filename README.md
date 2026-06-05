# EXAai-ADK

Standalone resume screening agent: parse resume and job description, enrich via Exa AI, score with Gemini, return JSON for the main hiring platform.

- **Plan:** [`implement.md`](./implement.md)
- **Pipeline flowcharts:** [`flowchart.md`](./flowchart.md)
- **ADK agent migration (phased):** [`docs/AGENT_MIGRATION.md`](./docs/AGENT_MIGRATION.md)
- **Architecture (ADK + Exa):** [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md)
- **Progress:** [`progress.md`](./progress.md)
- **Platform contract:** [`json-schema.md`](./json-schema.md) · [`CONTRACTS.md`](./CONTRACTS.md)

This service does **not** run on GCP. The main app (Supabase / Next.js) persists results.

## Requirements

- Python 3.12+
- API keys: [Google AI Studio](https://aistudio.google.com/apikey) (Gemini), [Exa](https://dashboard.exa.ai) (crawl)

## Setup

```bash
cd EXAai-ADK
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -e ".[dev]"
python -m spacy download en_core_web_sm
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

Edit `.env` and set at minimum `GEMINI_API_KEY`, `EXA_API_KEY`, and `API_KEYS`.

**Two different keys (easy to mix up):**

| `.env` variable | Purpose |
|-----------------|--------|
| `GEMINI_API_KEY` | Server → Google Gemini (never paste into Swagger) |
| `API_KEYS` | Clients → your `/screen` endpoint (paste into Swagger **api_key** field) |

**Default:** `SCREENING_MODE=agent` — ADK Runner; the agent picks profile URLs via tools. Set `SCREENING_MODE=pipeline` for the legacy enrich-all-then-score path.

**Fewer Gemini calls:** `JD_PARSE_USE_LLM=false` (default) uses local JD/resume parsing; `MAX_AGENT_TURNS=8` caps agent turns.

**OpenRouter (optional):** set `OPEN_ROUTER_API_KEY` and `LLM_PROVIDER=openrouter` (or `auto`). Default `OPENROUTER_MODEL_ID=openrouter/free` auto-picks a free model with tool-calling support; use `openai/gpt-oss-20b:free` for a specific fast model. On 429 rate limits, retries and `OPENROUTER_FALLBACK_MODEL_IDS` apply automatically.

## Run locally

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8080
```

Health check:

```bash
curl http://localhost:8080/health
```

## Call `/screen`

Multipart form request:

| Field | Required | Description |
|-------|----------|-------------|
| `application_id` | Yes | UUID from `job_applications` |
| `job_id` | Yes | UUID from `jobs` |
| `resume` | Yes | PDF or DOCX file (max 5 MB) |
| `jd` | Yes | JD file (PDF/DOCX/txt) or use `jd_text` |
| `jd_text` | No | Plain-text JD instead of file |

```bash
curl -X POST http://localhost:8080/screen \
  -H "Authorization: Bearer dev-local-key-change-me" \
  -H "X-Request-ID: $(uuidgen)" \
  -F "application_id=00000000-0000-0000-0000-000000000001" \
  -F "job_id=00000000-0000-0000-0000-000000000002" \
  -F "resume=@tests/fixtures/sample_resume.pdf" \
  -F "jd_text=Senior software engineer with Python and distributed systems."
```

Response shape: `resume-screening-result-v1` — see [`json-schema.md`](./json-schema.md). Migration phases: [`docs/AGENT_MIGRATION.md`](./docs/AGENT_MIGRATION.md).

## Tests

```bash
pytest
```

Output contract tests live in `tests/unit/test_validator.py`. JSON Schema source of truth: `agent/schema/resume-screening-result-v1.json`.

## Project layout

```
agent/     Pipeline, tools, security, schema, cache, audit
api/       FastAPI entrypoint
tests/     Unit and integration tests
```

## Main platform handoff

After a successful screen, the main app should call `POST /api/applications/update-score` with `resume_similarity_score` and set `resume_screening_status`. See [`CONTRACTS.md`](./CONTRACTS.md).
