# Project Context & AI Agent Map (AI.md)

> **Instructions for the AI:**
> Do not recursively scan this workspace. This file contains the architecture, data pathways, and directory layouts of the system. Reference this context map before generating or editing code files.

---

## 1. System Overview & Tech Stack
* **Project Name:** exaai-adk
* **Core Value Prop:** A standalone resume screening and verification service that parses resumes and job descriptions, redacts PII, extracts candidate portfolio URLs, sandboxes/evaluates GitHub repositories, crawls public profiles via Exa AI, and scores candidate fit using Gemini, producing structured, schema-validated JSON outputs for the main hiring platform.
* **Technology Stack:**
  * **Languages:** Python (>=3.12)
  * **Frameworks & Engines:** FastAPI, Uvicorn, Google ADK (Agent Development Kit), LiteLLM (multi-provider fallback)
  * **State & Data Stores:** In-process sessions (`InMemorySessionService` state), URL cache (`SQLite` in `./data/url_cache.db`), Screening result store (local JSON files in `./data/screening-results`)
  * **External APIs & Integrations:** Google Gemini API (default `gemini-2.0-flash` via `google-generativeai`), Exa AI API (via `exa-py`), GitHub API (via custom `GitHubClient`), Cloud Run / Docker (for repository execution sandboxing)

---

## 2. Pruned Codebase Map
*Only include files of critical structural or orchestration importance. Omit minor utilities, components, and tests to keep this map concise and highly informative.*

```text
C:/Users/Manav/Downloads/EXAai-ADK/
├── .env                             # Environment variables (GEMINI_API_KEY, EXA_API_KEY, API_KEYS)
├── CONTRACTS.md                     # Platform integration handoff contracts and status
├── README.md                        # Project overview, installation, and usage instructions
├── pyproject.toml                   # Project dependencies and tool configurations (hatchling backend)
├── agent/                           # Screening pipeline, security, and scoring logic
│   ├── adk_tools.py                 # ADK FunctionTools exposed to the screening agent
│   ├── agent_runner.py              # Screening agent instructions, nudge logic, and ADK runner loop
│   ├── config.py                    # Application settings loader using pydantic-settings
│   ├── deferred_screening.py        # Schedules deferred/background sandbox finalization
│   ├── enrichment.py                # Exa AI API wrapper & concurrent web crawling utilities
│   ├── pipeline.py                  # Orchestrator of screening flows (agent vs pipeline modes)
│   ├── prep.py                      # Parallel doc parsing, PII redaction, and links/rubric prep
│   ├── prep_context.py              # In-process session manager matching prep and runner states
│   ├── sandbox_gating.py            # Logic gate verifying if repo sandbox evaluation is required/done
│   ├── session_state.py             # Session state contract keys (SESSION_STATE_KEYS)
│   ├── submit.py                    # Screening submission process handler
│   ├── audit/
│   │   └── logger.py                # Writes audit logs for each screening run
│   ├── cache/
│   │   └── url_cache.py             # SQLite url_cache.db caching layer
│   ├── schema/                      # Schema files and Pydantic models for validation
│   │   ├── models.py                # Pydantic models representing output schema
│   │   ├── resume-screening-result-v1.json  # Platform schema source of truth
│   │   └── scoring-llm-response.json        # Scoring response format specification
│   ├── security/                    # Security layer
│   │   ├── allowlist.py             # Categorized domain allowlist for crawl URLs
│   │   ├── pii_redactor.py          # Microsoft Presidio/spaCy PII redactor
│   │   ├── profile_identity.py      # Profile identity verification and trust scoring
│   │   └── ssrf_guard.py            # SSRF validator checking URL length, hostname, and private IPs
│   └── tools/                       # Heuristics and deterministic scoring tools
│       ├── github_analyzer.py       # Orchestrates cloning and metadata extraction of candidate repos
│       ├── github_client.py         # Async GitHub client for fetching repo tree/languages
│       ├── link_extractor.py        # Extracts hyperlinks from parsed resume text
│       ├── parser.py                # Main parser for resumes and job descriptions (structured)
│       ├── portfolio_signal.py      # Evaluates candidate portfolio signals and penalties
│       ├── repo_focus.py            # Prepares file spec maps for sandbox evaluations
│       ├── repo_scoring.py          # Builds evaluation breakdown metrics
│       ├── result_sanitizer.py      # Utility sanitizers for scores, red flags, and metadata
│       ├── rubric_builder.py        # Generates job-rubric templates based on JD
│       ├── sandbox_prompt.py        # Formats sandbox output digests for scoring prompts
│       ├── sandbox_scoring.py       # Reconciles sandbox results and executes score capping
│       └── validator.py             # Detailed validator against platform JSON schema
├── api/                             # FastAPI application layer
│   ├── auth.py                      # ApiKey authentication check
│   ├── main.py                      # Application bootstrap & middleware registry
│   └── routes.py                    # Endpoint routes handler (/screen, /screenings/...)
```

---

## 3. Core Architectural flows & Rules

### System Data Flows

1. **Intake / Data Entry:** `/screen` endpoint (`api/routes.py`) accepts raw inputs (`resume` file, `jd` file/text, `application_id`, `job_id`) and validates formats.
2. **Preprocessing (Prep):** `prepare_screening_state` (`agent/prep.py`) runs concurrently: parses files, redacts PII, extracts hyperlinks, maps profile trust (`scoring_trusted`, `scoring_limited`, `scoring_untrusted`), and spawns a background thread to fetch GitHub repository metadata.
3. **Scoring / Agent Run:** `run_screening_async` (`agent/pipeline.py`) branches depending on configuration:
   - **Agent Mode (`SCREENING_MODE=agent`):** Launches an ADK `Runner` session (`agent/agent_runner.py`) using Gemini. The agent calls tools to inspect structures (`get_github_repo_structures`), fetch profiles (`fetch_profiles`), evaluate code files in sandboxes (`run_sandbox_analysis`), and submit findings (`submit_screening_result`).
   - **Pipeline Mode (`SCREENING_MODE=pipeline`):** Legacies flow. Concurrently fetches all allowlisted URLs via Exa, runs sandbox analysis, then scores in a single prompt call (`score_screening_from_state` in `agent/tools/scorer.py`).
4. **Post-Process & Handoff:** `normalize_screening_result` (`agent/tools/scorer.py`) applies deterministic score caps, checks must-haves, adds sandbox/portfolio penalties, formats outputs to the platform schema, writes audit logs, and returns the validated payload.

### Architectural Rules of Engagement

* **State Pattern:** Prep state snapshots are saved by `application_id` in `_PREP_BY_APPLICATION` (`agent/prep_context.py`) and merged at submission or fallback scoring to keep tool rounds lightweight and avoid token bloat.
* **Error Handling:** Schema validation errors return detailed lists (`errors` arrays). Failed runs must never crash the service; instead, they output structured failure objects via `build_failed_result` (`agent/tools/scorer.py`).
* **API Guardrails:** No raw external queries are permitted in API routers; they must go through the prep modules or ADK FunctionTools which enforce domain allowlists and SSRF guards.
* **SSRF and Allowlist Security:** Outbound crawls must be verified against `allowlist.py` (categorized domain prefixes) and `ssrf_guard.py` (checks hostname, length, scheme, resolves IPs, blocks localhost/private ranges).
* **Math & LLM Separation:** The LLM assigns individual rubric criterion scores (quantized to 5-point steps). The final `resume_similarity_score` must be computed deterministically using the weighted rubric mean `derive_overall_score_from_matches` in `rubric_builder.py` rather than letting the LLM compute or inflate it.
* **Deterministic Capping Rules:** Score caps must be enforced natively in python:
  * **Identity Cap:** Cap final score at 45 if any candidate URL is `scoring_untrusted` (famous handles/hijack risks).
  * **Must-Have Cap:** Cap final score at 45 if the candidate fails any must-have rubric item (score < 60).
  * **Sandbox Cap/Penalty:** Apply penalties and ceilings for vulnerabilities/secrets in aligned repositories.
  * **Portfolio Cap/Penalty:** Reduce score or cap at 75 if portfolio verification signals are missing or weak.

---

## 4. Token-Saving Guardrails for AI Engines

* **Scope Restriction:** Focus modifications and reviews only on the files mapped in Section 2. Request files explicitly via `view_file` rather than listing directories.
* **Output Brevity:** When implementing changes, generate isolated function overrides or precise diff blocks. Avoid reprinting long, unmodified files.
* **Dependency Guardrail:** Do not suggest importing new external libraries. Check `pyproject.toml` for existing packages and prioritize native standard libraries or registered tools.
