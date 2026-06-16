# EXAai-ADK

**AI-powered resume screening service** — parse resumes and job descriptions, enrich candidate profiles with Exa, score with Gemini (or other LLM providers), and return structured JSON for your hiring platform.

> This service runs **standalone** (not on GCP). Your main app (Supabase / Next.js) persists screening results.

---

## Features

| Capability | Description |
|------------|-------------|
| **Agent & pipeline modes** | ADK agent picks URLs via tools (`agent`) or legacy enrich-all-then-score (`pipeline`) |
| **JD + resume parsing** | PDF/DOCX extraction, local heuristics, optional Gemini structuring |
| **Profile enrichment** | Exa crawl with SSRF protection, allowlists, and trust scoring |
| **Rubric scoring** | Must-have / nice-to-have alignment with deterministic score derivation |
| **Portfolio verification** | Role-aware proof-of-work checks (GitHub, Kaggle, design portfolios, etc.) |
| **GitHub + sandbox** | Repo analysis, secret/vuln scanning, optional Cloud Run sandbox |
| **Structured output** | `resume-screening-result-v1` JSON contract with validation |

---

## Quick start

### Requirements

- Python **3.12+**
- API keys: [Google AI Studio](https://aistudio.google.com/apikey) (Gemini), [Exa](https://dashboard.exa.ai) (crawl)

### Install

```bash
git clone https://github.com/Manavv007/ExAai-SDK-resume-parser.git
cd ExAai-SDK-resume-parser
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
python -m spacy download en_core_web_sm
copy .env.example .env   # Windows — use cp on macOS/Linux
```

Edit `.env` and set at minimum:

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Server → Google Gemini (**do not** paste into Swagger) |
| `EXA_API_KEY` | Exa crawl API |
| `API_KEYS` | Clients → your `/screen` endpoint (Swagger **api_key** field) |

### Run

```powershell
# Recommended — reloads agent/ + api/ only
.\scripts\run_dev.ps1
```

```bash
./scripts/run_dev.sh
```

Manual:

```bash
uvicorn api.main:app --reload --reload-dir agent --reload-dir api \
  --timeout-graceful-shutdown 10 --host 0.0.0.0 --port 8080
```

Health check:

```bash
curl http://localhost:8080/health
```

---

## Screening flow

```mermaid
flowchart LR
    A[Resume + JD] --> B[Parse & structure]
    B --> C[Enrich profiles via Exa]
    C --> D[GitHub / sandbox optional]
    D --> E[LLM scoring + rubric]
    E --> F[Portfolio guardrails]
    F --> G[JSON result v1]
```

**Default mode:** `SCREENING_MODE=agent` — the ADK agent orchestrates URL selection and tool calls.  
Set `SCREENING_MODE=pipeline` for the legacy single-pass enrich-then-score path.

---

## Call `/screen`

`POST /screen` — multipart form:

| Field | Required | Description |
|-------|:--------:|-------------|
| `application_id` | ✓ | UUID from `job_applications` |
| `job_id` | ✓ | UUID from `jobs` |
| `resume` | ✓ | PDF or DOCX (max 5 MB) |
| `jd` | * | JD file (PDF/DOCX/txt) |
| `jd_text` | * | Plain-text JD instead of file |

*Provide `jd` **or** `jd_text`.*

```bash
curl -X POST http://localhost:8080/screen \
  -H "Authorization: Bearer dev-local-key-change-me" \
  -H "X-Request-ID: $(uuidgen)" \
  -F "application_id=00000000-0000-0000-0000-000000000001" \
  -F "job_id=00000000-0000-0000-0000-000000000002" \
  -F "resume=@tests/fixtures/sample_resume.pdf" \
  -F "jd_text=Senior software engineer with Python and distributed systems."
```

Response: `resume-screening-result-v1` — see [`json-schema.md`](./json-schema.md).

---

## Configuration highlights

| Setting | Default | Notes |
|---------|---------|-------|
| `SCREENING_MODE` | `agent` | `agent` or `pipeline` |
| `JD_PARSE_USE_LLM` | `false` | `false` saves one Gemini call per screen |
| `MAX_AGENT_TURNS` | `8` | Caps ADK agent tool turns |
| `LLM_PROVIDER` | `auto` | `gemini` · `groq` · `openrouter` · `auto` |
| `GITHUB_CLONE_ANALYSIS_ENABLED` | `false` | Enable repo clone + sandbox analysis |
| `AGENT_TRACE_ENABLED` | `false` | Structured JSON trace logs for agent/enrichment |
| `LOG_FORMAT` | `text` | Set `json` for machine-readable logs |

**OpenRouter (optional):** set `OPEN_ROUTER_API_KEY` and `LLM_PROVIDER=openrouter`. Agent mode needs tool calling — use `OPENROUTER_AGENT_MODEL_ID=openai/gpt-oss-20b:free`.

Full reference: [`.env.example`](./.env.example)

---

## Tests

```bash
pytest
ruff check agent api tests
```

Contract tests: `tests/unit/test_validator.py`  
JSON Schema: `agent/schema/resume-screening-result-v1.json`

---

## Project layout

```
agent/          Pipeline, ADK tools, scoring, security, sandbox
api/            FastAPI entrypoint (/screen, /health)
tests/          Unit and integration tests
docs/           Architecture, migration guides
```

---

## Documentation

| Doc | Description |
|-----|-------------|
| [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) | ADK + Exa architecture |
| [`docs/AGENT_MIGRATION.md`](./docs/AGENT_MIGRATION.md) | Phased agent migration |
| [`json-schema.md`](./json-schema.md) | Output schema reference |
| [`CONTRACTS.md`](./CONTRACTS.md) | Platform handoff contract |
| [`flowchart.md`](./flowchart.md) | Pipeline flowcharts |
| [`implement.md`](./implement.md) | Implementation plan |

---

## Main platform handoff

After a successful screen, the main app should call `POST /api/applications/update-score` with `resume_similarity_score` and set `resume_screening_status`. Details: [`CONTRACTS.md`](./CONTRACTS.md).
