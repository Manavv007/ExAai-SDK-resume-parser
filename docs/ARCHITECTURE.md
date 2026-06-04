# EXAai-ADK — Architecture (Google ADK + Exa)

## Your expected flow (yes — this is the target)

```text
JD + resume (files)
    → deterministic prep (parse, redact, link list, JD structured)
    → ADK Agent (Gemini) reads session context
    → LLM calls tools: fetch_profile_content(url) …  [Exa inside tool]
    → LLM produces resume-screening-result-v1 JSON
    → validate → return to main app
```

The **judging model** decides **which allowlisted URLs** are worth fetching. It does **not** call Exa directly — it calls our **ADK `FunctionTool`**, which runs SSRF + allowlist + `exa-py` + sanitization.

## What we built first (Phases 2–4)

Plain Python modules (`parser`, `link_extractor`, `pii_redactor`, etc.) so logic is **testable without the LLM**. They are **wrapped as ADK tools** — not thrown away.

## Two layers

| Layer | Technology | Role |
|-------|------------|------|
| **Prep** | Python (`agent/prep.py`) | Always runs first: parse PDF/DOCX, `jd_structured`, redact resume, extract URLs → `session.state` |
| **Screening agent** | `google-adk` `Agent` + `Runner` | Gemini orchestrates; calls tools; writes verdict JSON |

## ADK SDK usage

```python
from google.adk import Agent, Runner
from google.adk.tools import ToolContext  # alias of Context

screening_agent = Agent(
    model="gemini-2.0-flash",
    instruction="...",
    tools=[fetch_profile_content, list_candidate_profile_urls],
)
```

- **`Agent`** (`LlmAgent`): model with **function calling** (AutoFlow).
- **`FunctionTool`**: wraps Python functions; LLM chooses when to invoke.
- **`Runner`**: executes agent + session + events (wired in Phase 8 API).

`SequentialAgent` in older docs is **deprecated** in ADK 2.x; we use **`Agent` + tools** (and optional `Workflow` for purely deterministic graphs later).

## Security inside tools (non-negotiable)

Even when the LLM picks URLs, every `fetch_profile_content` call:

1. SSRF guard  
2. Domain allowlist  
3. Exa fetch (or cache, Phase 5)  
4. Sanitize + delimit content before returning to the model  

The model cannot bypass allowlist by wording alone.

## vs fully deterministic pipeline

| Approach | Pros | Cons |
|----------|------|------|
| **LLM + tools (chosen)** | Matches ADK; fetches only useful URLs; fewer Exa calls | Extra LLM turns; needs tool discipline |
| **Fixed order (old plan)** | Predictable cost/latency | Not using ADK orchestration; fetches all links |

We combine both: **prep is fixed**; **enrichment is LLM-driven via tools**.
