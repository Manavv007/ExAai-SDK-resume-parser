# Active Features & Modules: ExAai-ADK

This document provides a detailed breakdown of the functional modules and active capabilities implemented inside the ExAai-ADK screening service.

---

## 🔒 1. PII Redactor (`agent/security/pii_redactor.py`)

To protect candidate privacy and prevent screening biases (such as gender, origin, or location), the service redacts sensitive identifying details prior to passing content to the LLM:
* **Technology:** Built on **Microsoft Presidio Analyzer & Anonymizer** using a spaCy NLP engine (`en_core_web_sm`).
* **Target Entities:** Detects and redacts:
  * Person names (`<PERSON>`)
  * Email addresses (`<EMAIL_ADDRESS>`)
  * Phone numbers (`<PHONE_NUMBER>`)
  * Physical addresses and locations (`<LOCATION>`)
  * Dates and institutions (when matching contact headers)
* **Preservation of Length:** Returns a count of redacted items (`redaction_count`) to track redaction density while preserving general document layout.

---

## 📄 2. Document Parser & Link Extractor (`agent/tools/`)

Extracts textual data and hyperlinks from raw candidate documents:
* **File Formats:** Supports PDF, DOCX, and plain-text (`.txt`) resumes and job descriptions.
* **Hyperlink Extraction:** Extracts metadata links embedded directly inside PDF files (using `PyPDF2` / `pdfplumber` parsing equivalents) alongside plain-text URL patterns.
* **Platform Categorization:** Extracts candidate handles (GitHub, Kaggle, LinkedIn, Behance) and infers profile ownership matching using regex heuristics.

---

## 🌐 3. Exa AI Crawling & Content Caching (`agent/enrichment.py`)

Crawls allowed candidate websites to provide the screening agent with up-to-date portfolio data:
* **Exa AI Integration:** Performs semantic content crawls via Exa's REST endpoints (`exa-py` SDK).
* **Concurrency:** Fetches multiple profile URLs concurrently using Python `asyncio` and a configurable semaphore (`url_fetch_concurrency`, default: `12`).
* **Cache Layer:** Integrates with an SQLite cache (`url_cache.db`). If a URL was successfully fetched within the TTL (default: 24h), the crawler skips outbound network calls.
* **Content Sanitization:** Removes markup, boilerplate headers, script blocks, and potential prompt injection payloads from crawled text before feeding it to LLM contexts.

---

## 💻 4. Code Repository Sandboxing (`agent/sandbox/`)

Executes candidate code in a sandbox (Docker locally or GCP Cloud Run in production) to analyze code quality and verify portfolio authenticity:
* **Focus Mapping:** Builds repository file structure maps. Instead of scanning the entire project, the system selects the most critical files (e.g., core logic, configuration, entrypoints) for inspection (`top_file_evaluation`).
* **Vulnerability Scanning:** Executes diagnostic scanners inside the sandbox:
  * `pip-audit` / `npm audit` / `trivy` (scans package files for vulnerabilities).
  * Secrets scan (searches for exposed tokens, API keys, and environment passwords).
* **Execution Options:**
  * **Deferred Execution:** If `SANDBOX_DEFERRED_ENABLED=true`, a provisional result is returned immediately, and the sandbox runs in the background. The result status remains `processing` until updated by background tasks.
  * **Parallel Overlap:** If sandbox deferral is disabled but overlap is active, the sandbox runs concurrently with the agent scoring loop, joining before final submission.

---

## 📝 5. Rubric Builder (`agent/tools/rubric_builder.py`)

Converts job descriptions into a structured grading card:
* **Structural Categories:** Separates criteria into `technical_skill`, `soft_skill`, `experience`, `education`, and `responsibility`.
* **Must-Have vs. Nice-to-Have:** Flags required items that trigger score ceilings if unmet.
* **Bias-Avoidance Preamble:** Appends grading system rules to prompt contexts to enforce grading standards based solely on the candidate's actual qualifications.

---

## 🤖 6. ADK Screening Agent (`agent/agent_runner.py`)

Orchestrates URL selection, crawls, and sandbox execution using a Google ADK agent:
* **Agent Flow:** Gemini runs tool calls inside a session runner. The agent calls `list_candidate_profile_urls`, decides which allowlisted pages are relevant, calls `fetch_profiles` to crawl them, runs repository sandboxing on-demand, and structures findings.
* **Self-Correcting Submissions:** The agent submits results using `submit_screening_result`. If Pydantic validator checks fail, the tool returns detailed error messages. The agent is configured to read the error log, correct the JSON structure, and retry.
* **Turn Cap:** To control costs and execution time, the agent run loop is limited to a maximum number of turns (`MAX_AGENT_TURNS`, default: `8`).
