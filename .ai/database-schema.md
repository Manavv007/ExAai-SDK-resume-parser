# Database & Storage Schema: ExAai-ADK

This document outlines the state, caching, and result storage mechanisms used by the ExAai-ADK screening service, as well as the database schemas of the main hiring platform that integrates with it.

---

## 💾 1. Microservice Caching (SQLite: `url_cache.db`)

To optimize API usage and avoid repeatedly crawling external profile links, the service maintains a local SQLite database at `./data/url_cache.db` (configurable via `URL_CACHE_PATH`).

### Table: `url_cache`

| Column | Data Type | Key Constraints | Description |
| :--- | :--- | :--- | :--- |
| **`url_hash`** | `TEXT` | `PRIMARY KEY` | SHA-256 hash of the crawled URL. |
| **`url`** | `TEXT` | `NOT NULL` | The full original URL string. |
| **`content`** | `TEXT` | `NOT NULL` | Sanitized markdown/text content retrieved from the web page. |
| **`expires_at`** | `REAL` | `NOT NULL` | Epoch float timestamp representing when the cache entry expires. |

### Cache Policy:
* **TTL Duration:** Defaults to 24 hours (`86400` seconds), controlled by the `CACHE_TTL_SECONDS` configuration.
* **Upsert Behavior:** Uses `ON CONFLICT(url_hash) DO UPDATE SET content=excluded.content, expires_at=excluded.expires_at` to refresh active cache entries.
* **Eviction:** Stale records are lazily deleted during lookups if their `expires_at` timestamp is in the past.

---

## 📁 2. Local Screening Storage (`ScreeningResultStore`)

The service stores temporary JSON records under `./data/screening-results/` (configurable via `SCREENING_RESULT_STORE_PATH`). These files provide immediate state access for client polling.

### File Naming Convention:
```text
./data/screening-results/<application_id>__<job_id>.json
```

### JSON Envelope Schema:

```json
{
  "application_id": "00000000-0000-0000-0000-000000000001",
  "job_id": "00000000-0000-0000-0000-000000000002",
  "status": "completed",
  "updated_at": "2026-06-21T15:02:44.123Z",
  "result": {
    "application_id": "00000000-0000-0000-0000-000000000001",
    "job_id": "00000000-0000-0000-0000-000000000002",
    "resume_screening_status": "completed",
    "resume_similarity_score": {
      "score": 85,
      "reasoning": "Strong engineering profile matching qualifications."
    },
    "requirement_matches": [
      {
        "requirement": "Python distributed systems development",
        "requirement_type": "technical_skill",
        "match_score": 90,
        "evidence": "Candidate built async API frameworks handling parallel tasks.",
        "source_quote": "Lead Python Engineer building highly concurrent backends."
      }
    ],
    "recommendation": "advance",
    "recommendation_reasoning": "Strong match with excellent portfolio quality.",
    "sources_crawled": [
      {
        "url": "https://github.com/candidate-username",
        "relevance": "high",
        "title": "GitHub Profile"
      }
    ],
    "metadata": {
      "schema_version": "1.0",
      "model_version": "gemini-2.0-flash",
      "processed_at": "2026-06-21T15:02:44.123Z",
      "resume_text_chars": 4500,
      "agent_version": "0.1.0"
    },
    "errors": []
  }
}
```

---

## 🏛️ 3. Main Hiring Platform DB Schema (Next.js / Supabase)

After completing a screening run, the integration client writes outputs back to the primary PostgreSQL database on Supabase.

### Table: `jobs`
Holds job post definitions.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| **`job_id`** | `UUID` | `PRIMARY KEY`, default: `gen_random_uuid()` | Unique job identifier. |
| **`job_title`** | `TEXT` | `NOT NULL` | The title of the role. |
| **`initial_job_desc_json`** | `JSONB` | | Original unstructured input job description. |
| **`final_job_desc_json`** | `JSONB` | | Structured job details schema containing responsibilities, qualifications, and criteria. |
| **`created_at`** | `TIMESTAMPTZ`| `NOT NULL`, default: `now()` | Creation timestamp. |
| **`updated_at`** | `TIMESTAMPTZ`| `NOT NULL`, default: `now()` | Last update timestamp. |

### Table: `job_applications`
Persists application files, status values, and LLM similarity scores.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| **`application_id`** | `UUID` | `PRIMARY KEY`, default: `gen_random_uuid()` | Unique application identifier. |
| **`job_id`** | `UUID` | `REFERENCES jobs(job_id)` | Linked job reference. |
| **`candidate_user_id`** | `UUID` | `REFERENCES user_profiles(user_id)` | Linked candidate profile reference. |
| **`resume_url`** | `TEXT` | `NOT NULL` | URL pointing to the raw uploaded PDF/DOCX file in storage. |
| **`resume_screening_status`** | `TEXT` | | Status: `queued` \| `processing` \| `completed` \| `failed`. |
| **`resume_similarity_score`** | `JSONB` | | Score summary containing: `{ score: number, reasoning: string }`. |

### Table: `user_profiles` (Candidates)
Defines basic candidate contact and bio details.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| **`user_id`** | `UUID` | `PRIMARY KEY` (matches `auth.users.id`) | Primary profile identifier. |
| **`user_email`** | `TEXT` | `NOT NULL` | Contact email address. |
| **`resumes`** | `JSONB` | | Array of JSON structures tracking candidate resume history: `[{url, filename, uploaded_at}]`. |
| **`experience`** | `JSONB` | | Structured experience resume blocks. |
