# Project Overview: ExAai-ADK

Welcome to **ExAai-ADK**, an AI-powered resume screening and candidate profile enrichment service. This microservice parses resume files and job descriptions, automatically redacts Personally Identifiable Information (PII), extracts public profile URLs, crawls target pages via Exa AI, evaluates GitHub repositories in secure execution sandboxes, and scores candidates against a job rubric using Gemini.

---

## 🚀 Core Value Proposition

In modern recruiting, resumes only tell half the story. ExAai-ADK provides deep screening by verifying candidates' real-world portfolios and public presence:
1. **Automated Verification:** Rather than taking resume claims at face value, the system crawls and cross-references candidate links (GitHub, LinkedIn, Kaggle, personal portfolios) via **Exa AI**.
2. **Execution Sandboxing:** Code repositories are cloned and analyzed in a secure execution sandbox (e.g., Google Cloud Run or Docker) to verify technical validity, scan for vulnerabilities/secrets, and evaluate code quality.
3. **Structured & Fair Scoring:** Screening criteria (technical skills, soft skills, experience, education, responsibilities) are structured into a detailed rubric. Match scoring is done by Gemini, but final overall scoring and caps are calculated deterministically via Python mathematical rules.
4. **State-of-the-Art PII Protection:** The system automatically redacts sensitive data (names, emails, phones, addresses) before sending resume content to LLMs, ensuring compliance with data privacy regulations.

---

## 🛠️ Technology Stack

The service is built as a lightweight, high-performance Python application:

| Layer | Component / Tool | Role |
| :--- | :--- | :--- |
| **Core Runtime** | **Python (>=3.12)** | Main application runtime environment. |
| **API Framework** | **FastAPI & Uvicorn** | Fast, async web framework for API routing and Swagger UI generation. |
| **AI Orchestration** | **Google ADK (Agent Development Kit)** | Framework for defining the screening agent, FunctionTools, and LLM runner. |
| **LLM Clients** | **LiteLLM / Google GenAI SDK** | Multi-provider LLM connector (supporting Gemini, Groq, OpenRouter). |
| **Information Extraction** | **Microsoft Presidio & spaCy** | AI-driven PII identification and text redaction. |
| **Web Crawling** | **Exa AI (`exa-py`)** | Neural search and crawl API to retrieve candidate public profile contents. |
| **Secure Sandbox** | **GCP Cloud Run Jobs / Docker** | Execution environments for sandboxing candidate repository code. |
| **State & Caching** | **SQLite & In-Memory Sessions** | Local TTL cache (`url_cache.db`) and local screening result persistence. |

---

## 🔄 System Architecture at a Glance

The following diagram highlights how data flows from intake (`POST /screen`) to the finalized structured JSON payload:

```mermaid
flowchart TD
    Start[POST /screen request] --> Parse[1. Parser: Extract text from PDF/DOCX]
    Parse --> Redact[2. PII Redactor: Redact names, phone, email]
    Redact --> LinkExt[3. Link Extractor: Extract GitHub, LinkedIn, Kaggle URLs]
    LinkExt --> TrustMap[4. Trust Profiler: Score link ownership & security risk]
    TrustMap --> GithubStart[5. Start Background GitHub Prep Thread]
    
    subgraph Execution Loop (Agent Mode)
        AgentInit[6. Seed ADK Session State] --> ListUrls[list_candidate_profile_urls]
        ListUrls --> FetchProfiles[fetch_profiles via Exa]
        FetchProfiles --> RepoStructures[get_github_repo_structures]
        RepoStructures --> RunSandbox[run_sandbox_analysis]
        RunSandbox --> SubmitResult[submit_screening_result]
    end
    
    GithubStart --> AgentInit
    SubmitResult --> PostProcess[7. Post-Process: Apply Math Caps & Portfolio Penalties]
    PostProcess --> Validator[8. Pydantic Validator: Match JSON Schema v1.0]
    Validator --> Success[9. Persist JSON & return 200 OK]
    Validator -- Errors --> Retry[10. Self-Correcting LLM Retry]
    Retry --> SubmitResult
```

---

## 📂 Project Directory Structure

```text
EXAai-ADK/
├── .ai/                             # Project documentation (this folder)
├── .env                             # Active environment variable settings
├── pyproject.toml                   # Project dependencies and tool configurations
├── README.md                        # Quick start, installation, and manual usage instructions
├── CONTRACTS.md                     # Platform integration handoff contracts and status
├── json-schema.md                   # Detailed output JSON contract and Swagger reference
├── agent/                           # Main logic directory
│   ├── audit/                       # Structured audit logging
│   ├── cache/                       # SQLite cache layer for crawled URLs
│   ├── schema/                      # Output JSON schema files and Pydantic models
│   ├── security/                    # PII redaction, SSRF guard, allowlist, profile identity trust
│   ├── tools/                       # Heuristic analysis, rubric builder, sandbox, validator
│   ├── adk_tools.py                 # google-adk FunctionTool wrappers
│   ├── agent_runner.py              # Screening agent instruction prompts and runner loops
│   ├── config.py                    # Application configuration loader (pydantic-settings)
│   ├── pipeline.py                  # Core pipelines (Agent mode vs Legacy Pipeline mode)
│   ├── prep.py                      # Preprocessing logic (parsing, PII, links, trust mapping)
│   └── session_state.py             # ADK Session State contract keys
├── api/                             # FastAPI application layer
│   ├── auth.py                      # Bearer token validation
│   ├── routes.py                    # API route endpoints (/screen, /screenings/...)
│   └── main.py                      # FastAPI app setup and middleware routing
└── tests/                           # Unit and integration test suites
```

---

## 🔧 Production Configuration Settings

You can customize the screening pipeline behavior by adjusting variables in your `.env` file. The primary settings include:

* **`SCREENING_MODE`**: Options are `agent` (default, uses the ADK agent) and `pipeline` (runs a single-pass fallback scoring flow).
* **`LLM_PROVIDER`**: Set to `gemini` (default), `groq`, `openrouter`, or `auto` (detects keys).
* **`GITHUB_CLONE_ANALYSIS_ENABLED`**: Enables cloning of candidate code repositories for sandboxing (`true`, `false`, or `auto`).
* **`SANDBOX_DEFERRED_ENABLED`**: When `true`, returns a provisional score immediately (`processing` status) and finalizes sandbox checks in background tasks.
* **`MAX_URLS_PER_RESUME`**: Limits the number of extracted candidate URLs (default: `10`).

For a complete reference of configurations, see the settings schema in [config.py](file:///C:/Users/Manav/Downloads/EXAai-ADK/agent/config.py).
