# ADK single-agent migration plan

Move from **pipeline** (`enrich all URLs` → `score_screening`) to **Option A**: one Gemini ADK agent that calls Exa via tools and submits the final verdict via `submit_screening_result`.

**You approve each phase before we implement it.**

---

## Target architecture

```text
POST /screen
    │
    ▼
PREP (unchanged, Python)
  parse, redact, jd_structured, rubric, profile_trust, profile_urls
    │
    ▼
ADK Runner + screening Agent (Gemini)
  • list_candidate_profile_urls
  • fetch_profiles(urls[])     ← parallel Exa inside tool
  • submit_screening_result()  ← validator in tool; agent retries on error
    │
    ▼
POST-PROCESS (unchanged, Python)
  must-have cap, identity cap, merge red flags, audit log
    │
    ▼
resume-screening-result-v1 JSON
```

**Prep and post-process stay deterministic.** Only enrichment + scoring move under the agent.

---

## Agent instruction (final)

```text
You are a resume screening agent.

1. Read session: redacted resume, job description, rubric, profile_trust_by_url.
2. Call list_candidate_profile_urls to see allowlisted URLs and trust tiers.
3. Fetch only URLs that materially help assess JD fit. Prefer GitHub, portfolio
   sites, and Kaggle. Stop when resume plus fetched evidence is enough.
   Do NOT fetch scoring_untrusted URLs. You may call fetch_profiles(urls) again
   until the session fetch budget (max_urls_per_resume) is reached.
4. Base requirement evidence on the redacted resume first; use fetched content
   only for scoring_trusted (or corroborated limited) profiles.
5. When ready, call submit_screening_result with resume-screening-result-v1 JSON.
   If the tool returns validation errors, fix and resubmit.
```

---

## Phase 1 — Tools foundation (`fetch_profiles`)

**Goal:** Batch parallel fetch tool; keep existing single-URL tool for compatibility.

| Task | File(s) |
|------|---------|
| Add `fetch_profiles(urls: list[str], tool_context)` | `agent/adk_tools.py` |
| Reuse `fetch_profile_url` per URL inside `asyncio.gather` (session cap) | `agent/enrichment.py` |
| Reject URLs not in `profile_urls` or `scoring_untrusted` | `agent/adk_tools.py` |
| Return per-URL status + short preview (not full body in tool response) | `agent/adk_tools.py` |
| Unit tests: mock Exa, trust rejection, session budget | `tests/unit/test_adk_tools.py` |

**Done when:** Agent can call `fetch_profiles(["url1","url2"])` in isolation tests; pipeline `/screen` still uses old path.

**Acceptance criteria:**
- [x] Untrusted URL in list → skipped with reason, not fetched
- [x] Session fetch budget exceeded → truncate (`truncated` count in response)
- [x] Parallel fetch uses same SSRF/allowlist/sanitize path as today

---

## Phase 2 — `submit_screening_result` tool

**Goal:** Agent must submit JSON through a tool; validator runs inside tool.

| Task | File(s) |
|------|---------|
| Add `submit_screening_result(result: dict, tool_context)` | `agent/adk_tools.py` |
| Run `validate_result_detailed` + UUID checks | `agent/tools/validator.py` |
| On success: store `state["screening_result"]`, return `{ok: true}` | |
| On failure: return `{ok: false, errors: [...]}` for agent retry | |
| Apply `normalize_screening_result` (caps, identity flags) before validate | `agent/tools/scorer.py` |
| Unit tests: valid fixture passes; bad UUID / missing field fails | `tests/unit/test_adk_tools.py` |

**Done when:** Tool accepts `valid_result_completed.json` shape and rejects malformed payloads with clear errors.

**Acceptance criteria:**
- [x] Valid completed result → `state["screening_result"]` set
- [x] Invalid JSON fields → tool error message agent can read
- [x] Identity cap + must-have cap applied before validate

---

## Phase 3 — Agent definition and session wiring

**Goal:** Single agent with all tools and rich instruction; not yet default in API.

| Task | File(s) |
|------|---------|
| Expand `create_screening_agent()` instruction (see above) | `agent/pipeline.py` |
| Register tools: `list_candidate_profile_urls`, `fetch_profiles`, `submit_screening_result` | `agent/pipeline.py` |
| Optional: deprecate `fetch_profile_content` or keep as single-URL alias | `agent/adk_tools.py` |
| Build initial user message from state (JD, resume, rubric summary, trust map) | `agent/agent_runner.py` (new) |
| Document session keys for agent path | `agent/session_state.py` |
| Unit test: agent has 3+ tools registered | `tests/unit/test_pipeline.py` |

**Done when:** `create_screening_agent().tools` lists all tools; instruction includes trust + session fetch budget.

**Acceptance criteria:**
- [x] Agent model from `GEMINI_MODEL_ID`
- [x] Instruction references `profile_trust_by_url` behavior

---

## Phase 4 — `run_screening_agent_async` (Runner integration)

**Goal:** Execute agent loop end-to-end; read result from `state["screening_result"]`.

| Task | File(s) |
|------|---------|
| New `run_screening_agent_async(state)` | `agent/agent_runner.py` |
| Use `create_runner()` + session seeded from prep state | `agent/pipeline.py` |
| Max agent turns / tool rounds config (e.g. `MAX_AGENT_TURNS=8`) | `agent/config.py` |
| Handle: no submit → `build_failed_result(LLM_ERROR)` | |
| Handle: timeout / empty response | |
| Integration test: mocked tools or mocked Gemini tool-calls | `tests/integration/test_agent_screening.py` |

**Done when:** Full prep → agent (mocked LLM) → `screening_result` in tests.

**Acceptance criteria:**
- [x] Successful mock agent run → `resume_screening_status: completed`
- [x] Agent never calls Exa outside tools (mock asserts)

---

## Phase 5 — Feature flag and API switch

**Goal:** Production can choose pipeline vs agent; default to agent when ready.

| Task | File(s) |
|------|---------|
| `SCREENING_MODE=pipeline|agent` in config + `.env.example` | `agent/config.py` |
| `run_screening_async` branches on mode | `agent/pipeline.py` |
| Default `agent` after Phase 5 sign-off (or keep `pipeline` until you say) | |
| Update `flowchart.md` with agent path | |
| Manual smoke: `/docs` one real screen in agent mode | |

**Done when:** `.env` flip switches behavior; both paths return same schema.

**Acceptance criteria:**
- [x] `SCREENING_MODE=agent` uses Runner
- [x] `SCREENING_MODE=pipeline` unchanged (fallback)
- [x] Same post-process caps on both paths

---

## Phase 6 — Hardening, tests, cleanup

**Goal:** CI green; pipeline path optional/deprecated.

| Task | File(s) |
|------|---------|
| Integration: injection not in evidence (agent path) | `tests/integration/test_security_hardening.py` |
| Integration: identity untrusted → agent instructed not to fetch | |
| Domain fixtures (software/design/academic) agent mode | `tests/integration/test_domain_agent.py` |
| README + ARCHITECTURE.md reflect agent as primary | |
| Optional: remove parallel `enrich_profile_urls_async` from default path only | |
| `progress.md` Phase 12 agent migration checkboxes | |

**Done when:** 121+ tests pass; you sign off agent mode as default.

- [x] Agent-path injection test (`test_agent_sanitizes_injection_before_submit`)
- [x] Agent-path untrusted URL skips Exa (`test_agent_skips_untrusted_profile_fetch`)
- [x] Domain fixtures in agent mode (`tests/integration/test_domain_agent.py`)
- [x] README + `ARCHITECTURE.md` reflect agent as primary
- [x] `progress.md` Phase 12 checkboxes
- [x] Default `SCREENING_MODE=agent` in config + `.env.example`

---

## Config additions (all phases)

```env
SCREENING_MODE=agent          # agent (default) | pipeline
MAX_AGENT_TURNS=8
AGENT_RUN_TIMEOUT_SECONDS=120
MAX_URLS_PER_RESUME=10        # resume URL cap + agent session fetch budget
```

---

## What stays pipeline-style forever

| Step | Why |
|------|-----|
| `prepare_screening_state` | Parsing, PII, rubric, identity trust |
| SSRF + allowlist in tools | Security |
| `normalize` + caps after submit | Deterministic fairness |
| `validate_result_detailed` | Contract with main app |
| Audit log | Compliance |

---

## Rollback

Set `SCREENING_MODE=pipeline` in `.env` and restart uvicorn — no code deploy needed.

---

## Phase status

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | `fetch_profiles` batch tool | ✅ Done |
| **2** | `submit_screening_result` tool | ✅ Done |
| **3** | Agent + instruction + session | ✅ Done |
| **4** | Runner `run_screening_agent_async` | ✅ Done |
| **5** | Feature flag + API switch | ✅ Done |
| **6** | Tests, docs, default agent | ✅ Done |

_Update this table as each phase completes._

---

## How to proceed

Reply with e.g. **"Start Phase 1"** and we implement only that phase, run tests, and update the status table before moving on.
