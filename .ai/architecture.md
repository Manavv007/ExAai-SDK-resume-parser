# System Architecture: ExAai-ADK

This document describes the high-level architecture, orchestration flow, and security mechanisms of the ExAai-ADK resume screening service.

---

## 🏗️ Core Architecture Overview

The system is designed with a **two-layer processing pipeline**:
1. **Deterministic Prep Layer (Python):** Prepares state, parses files, redacts PII, scores link trust levels, structures job criteria, and initiates repository checkouts asynchronously.
2. **AI Screening Layer (ADK / LLM):** Orchestrated via the **Google Agent Development Kit (ADK)**. The screening agent dynamically calls tools to discover more about the candidate and yields a finalized, schema-validated screening JSON.

```text
                               ┌───────────────────────────────────┐
                               │       Client (FastAPI /screen)     │
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
     ┌───────────────────────────────────────────────────────────────────────────────────────┐
     │                             1. DETERMINISTIC PREP LAYER                               │
     │                                                                                       │
     │   ┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌─────────────┐   │
     │   │  Doc Parser   │     │ PII Redactor  │     │ Link Extractor│     │ Trust Map   │   │
     │   │  (PDF/DOCX)   │     │  (Presidio)   │     │ (Hyperlinks)  │     │ (Allowlist) │   │
     │   └───────┬───────┘     └───────┬───────┘     └───────┬───────┘     └──────┬──────┘   │
     │           └─────────────────────┼─────────────────────┼────────────────────┘          │
     │                                 ▼                                                     │
     │                 [ Session Prep State Seeded in Memory ]                               │
     │                                 │                                                     │
     │                                 ▼                                                     │
     │                 ( Background Thread: Clone & Prep Repo )                              │
     └─────────────────────────────────┬─────────────────────────────────────────────────────┘
                                       │
                                       ▼
     ┌───────────────────────────────────────────────────────────────────────────────────────┐
     │                           2. ADK AGENT SCREENING LAYER                                │
     │                                                                                       │
     │   [ ADK Runner Session ]                                                              │
     │      ├─► list_candidate_profile_urls() -> (Trust & Identity categorization)           │
     │      ├─► fetch_profiles(urls[])        -> (SSRF, Allowlist, SQLite Cached Exa fetch)  │
     │      ├─► get_github_repo_structures()   -> (Repo files & focus map)                   │
     │      ├─► run_sandbox_analysis()         -> (Gating checks, execution & report logs)    │
     │      └─► submit_screening_result(JSON)  -> (Trigger Scoring normalization & validate) │
     └─────────────────────────────────┬─────────────────────────────────────────────────────┘
                                       │
                                       ▼
     ┌───────────────────────────────────────────────────────────────────────────────────────┐
     │                           3. POST-PROCESSING & VALIDATION                             │
     │                                                                                       │
     │   [ Scoring Rules Engine ]                                                            │
     │      ├─► Must-Have Cap check (fails rubric item -> cap score at 45)                   │
     │      ├─► Identity Cap check (contains untrusted profile hijack -> cap score at 45)    │
     │      ├─► Sandbox vulnerability penalty ceilings applied deterministically             │
     │      ├─► Pydantic validator checks schema constraints                                 │
     │      └─► Persist completed JSON payload to /data/screening-results/                   │
     └───────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📥 1. Deterministic Prep Layer (`agent/prep.py`)

When `/screen` receives a request, `prepare_screening_state()` initializes state. Since several tasks are CPU-intensive or require external network pre-steps, **concurrency is maximized using a `ThreadPoolExecutor`**:

1. **Document Extraction:** Parses PDF/DOCX resumes and JDs. Discovers hyperlinks embedded in PDF metadata.
2. **Concurrent Sub-Tasks:**
   * **PII Redactor:** Identifies sensitive fields and redacts them, generating a character length count.
   * **Link Extractor:** Detects plain-text and metadata URLs, categorizing them by domain (e.g., GitHub, Kaggle).
   * **Structured JD Parser:** Builds a JSON representation of job criteria (skills, years of experience, education).
3. **Background GitHub Initialization:** If candidate GitHub repositories are found, a background thread is spawned to fetch metadata and setup sandbox jobs immediately. This prevents the API from blocking during initial network lookup.
4. **Rubric Builder:** Generates criteria evaluation cards containing weight specifications.
5. **Link Identity Trust Profiler:** Inspects links to build a trust matrix:
   * **`scoring_trusted`:** Direct links matching candidate name patterns.
   * **`scoring_limited`:** Uncorroborated profiles (need additional checking).
   * **`scoring_untrusted`:** Famous handles (high identity theft/hijack risk, blocked from fetch).

---

## 🤖 2. ADK Agent Screening Layer (`agent/pipeline.py`)

If `SCREENING_MODE=agent` (default), the Google ADK runner spawns the `resume_screener` agent. The agent executes a loop using Gemini and utilizes specific tools:

| FunctionTool Name | Description |
| :--- | :--- |
| `list_candidate_profile_urls` | Returns candidate URLs, domain categories, and trust classifications. |
| `fetch_profiles` | Crawls allowed profiles via Exa AI. Resolves from the SQLite cache when possible. |
| `get_github_repo_structures` | Discovers repository trees and maps structural layout. |
| `run_sandbox_analysis` | Triggers a sandboxed evaluation of candidate code and returns diagnostic metrics. |
| `submit_screening_result` | Validates the screening output format. On schema error, returns feedback to the agent for a retry. |

---

## 🔒 3. Outbound Security Architecture (`agent/security/`)

Outbound connections inside the Exa crawling tool and sandboxing modules must follow strict security constraints to prevent infrastructure vulnerability:

### A. SSRF Guard (`agent/security/ssrf_guard.py`)
To prevent Server-Side Request Forgery (SSRF), all crawl URLs pass through `validate_url()`:
* **HTTPS Only:** Rejects plain `http` or ftp schemes.
* **URL Length Limit:** Maximum length capped at 2048 characters.
* **No IP Hostnames:** Blocks raw IPv4/IPv6 hostnames (e.g. `https://127.0.0.1/`).
* **DNS Resolution Resolution:** Resolves hostnames using `socket.getaddrinfo` (cached in-memory for 60 seconds).
* **Local/Private IP Block:** Blocks RFC 1918 private ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), loopback (`127.0.0.0/8`), link-local (`169.254.0.0/16`), and reserved space.

### B. Domain Allowlist (`agent/security/allowlist.py`)
Outbound crawling is locked down to a hardcoded set of trusted web categories. The allowlist contains specific domains under:
* **Code repositories:** `github.com`, `gitlab.com`, `bitbucket.org`, `kaggle.com`.
* **Portfolios:** `behance.net`, `dribbble.com`, `webflow.io`, `vercel.app`.
* **Professional:** `linkedin.com`, `wellfound.com`.
* **Academic/Writing:** `scholar.google.com`, `arxiv.org`, `dev.to`, `substack.com`.

---

## 📦 4. Execution Sandbox Mechanics (`agent/sandbox_gating.py`)

Candidate repositories are run through a Docker sandbox (locally) or GCP Cloud Run Jobs (in production).
* **Pre-run vs. On-Demand:** Based on settings, the sandbox can execute during prep time, in parallel with LLM loops, or on-demand when the agent requests it via `run_sandbox_analysis`.
* **Vulnerability Scanning:** Checks files for hardcoded secrets, database credentials, and vulnerabilities.
* **Provisional Scoring / Deferred Mode:** If `SANDBOX_DEFERRED_ENABLED=true`, the pipeline returns a temporary score and a status of `processing` to avoid blocking the client request (e.g. for long-running test execution). The evaluation is completed in background tasks and the final JSON is updated once done.
