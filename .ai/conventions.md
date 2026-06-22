# Coding & Scoring Conventions: ExAai-ADK

This document lists the formatting patterns, data constraints, validation rules, and scoring mathematical logic enforced across the ExAai-ADK service.

---

## 🏷️ 1. Naming & Data Type Conventions

To maintain strict interoperability with the primary platform (Next.js + Supabase + TypeScript), the service follows these schema naming rules:

* **Case Sensitivity:**
  * **TypeScript Layer (UI):** Uses `camelCase` (e.g. `resumeSimilarityScore`, `applicationId`, `processedAt`).
  * **Database Columns (Postgres & SQLite):** Uses `snake_case` (e.g. `resume_similarity_score`, `application_id`, `expires_at`).
  * **Pydantic Serialization:** Handled in [models.py](file:///C:/Users/Manav/Downloads/EXAai-ADK/agent/schema/models.py) to parse inputs as snake_case and match the platform database schema.
* **Identifiers:** Always passed and validated as raw **UUID strings** (UUIDv4) for `application_id` and `job_id`.
* **Timestamps:** Always serialized as **ISO 8601 UTC strings** (e.g. `2026-06-21T15:02:44.123Z`).

---

## 🔒 2. Profile Verification Trust Classification

When parsing hyperlinks from resumes and social blocks, the identity engine resolves links into distinct trust buckets, determining how heavily the agent weighs their content:

| Trust Tier | Meaning | Fetch Rule |
| :--- | :--- | :--- |
| **`scoring_trusted`** | High confidence of candidate ownership (e.g., links directly mapping to candidate name patterns). | Allowed to fetch. Content is fully integrated into evidence. |
| **`scoring_limited`** | Low confidence or uncorroborated profile domain. | Allowed to fetch. Content is integrated but tagged with a warning. |
| **`scoring_untrusted`** | High risk of identity theft, squatted names, or famous profiles (e.g., handles like `linus-torvalds`). | **BLOCKED** from fetching. Exa crawler tool will ignore these URLs. |

---

## 📊 3. Rubric & Quantized Scoring

* **Quantized Scores:** Requirement scores (`match_score`) and overall scores are quantized using a step size of **5** (e.g. `60`, `65`, `70`, `75`, `80`...). This avoids arbitrary point noise.
* **Separation of LLM & Math:**
  * The LLM's role is to evaluate specific requirements in isolation, assigning a score of `0-100` and extracting quotes as evidence.
  * The LLM **does not calculate the final score**.
  * The overall `resume_similarity_score` is computed deterministically in Python using the weighted rubric average (`derive_overall_score_from_matches` in `rubric_builder.py`). This prevents scoring inflation and bias.

---

## 🛑 4. Deterministic Score Capping Rules

After initial scoring, mathematical rules are applied to adjust the final score. These caps are non-negotiable:

### A. Identity Trust Cap
* **Trigger:** The candidate has one or more profile URLs resolved as `scoring_untrusted` (high hijack or fake profile risk).
* **Action:** The final score is capped at a maximum of **45**.

### B. Must-Have Rubric Cap
* **Trigger:** The candidate fails to meet any rubric requirement marked as **must-have** (scored below **60**).
* **Action:** The final score is capped at a maximum of **45**.

### C. Execution Sandbox Penalties & Ceilings
When repositories are run in a secure sandbox, security and quality findings adjust the score:
* **Risk Penalty:** Subtractions are calculated for each repository:
  * Weak secret hygiene (e.g., hardcoded credentials): `-4` points.
  * High-severity findings: `-2` points per finding.
  * Vulnerability count (Trivy/pip-audit/npm): Capped penalties (`-2` to `-8` points depending on quantity).
  * Blended aggregate: Penalties are blended using a formula weighting the worst repository (`70% worst + 30% average`), up to a maximum total penalty of `10` points.
* **Risk Score Ceiling:** If an aligned repository is highly active but exhibits severe risk, hard ceilings are applied to prevent the candidate from bypassing safety checks:
  * Aligned repo with `vulnerability_count >= 100` or `secrets found + 20+ vulnerabilities`: Max score ceiling of **85**.
  * Aligned repo with `50+ vulnerabilities`: Max score ceiling of **88**.

### D. Portfolio Cap
* **Trigger:** For engineering roles, if portfolio signals are missing or untrusted.
* **Action:** The final score is capped at a maximum of **75**.
