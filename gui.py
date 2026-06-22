"""Local Gradio GUI: preprocess like a pipeline, then run the DEPLOYED bare agent.

This mirrors what run_screening_agent_async does locally, but against a deployed
AdkApp (chat-style) agent:

  1. Fetch the resume bytes from the URL.
  2. Run prepare_screening_state() LOCALLY -> the same prep state the pipeline builds
     (parsed/redacted resume, profile_urls, jd_structured, rubric, trust map).
  3. Create a remote agent session SEEDED with that prep state, so the agent's tools
     (fetch_profiles, classify_portfolio_role, submit_screening_result) can read it.
  4. stream_query the agent with the prep "brief" (JD + redacted resume + rubric) —
     the LLM scores from the brief; the tools use the seeded state.
  5. Read screening_result back from the session state.

Auth uses Application Default Credentials (run: gcloud auth application-default login),
so there is no endpoint access-token field. The resume Bearer token is only used to
fetch an authenticated https:// resume URL.

Caveat: the bare agent has no post-agent robustness layer (continuation nudges,
pipeline rescore) — that lives in run_screening_agent_async. For the full, robust
pipeline server-side, deploy ResumeScreeningApp and call screen() instead.

Run:
  python gui.py
"""

from __future__ import annotations

import asyncio
import json
import uuid

import gradio as gr
import vertexai
from dotenv import load_dotenv

from agent.agent_engine_app import ResumeScreeningApp
from agent.agent_runner import build_agent_user_message
from agent.enrichment import enrich_profile_urls_async
from agent.prep import prepare_screening_state

load_dotenv()

PROJECT_ID = "serin-490413"
LOCATION = "us-central1"
DEFAULT_AGENT_RESOURCE = (
    "projects/127610777818/locations/us-central1/reasoningEngines/4666246534176702464"
)


def _session_id(session: object) -> str | None:
    if isinstance(session, dict):
        return session.get("id") or session.get("session_id")
    return getattr(session, "id", None) or getattr(session, "session_id", None)


def _session_state(session: object) -> dict:
    if isinstance(session, dict):
        return session.get("state") or {}
    return getattr(session, "state", {}) or {}


def run_pipeline_then_agent(
    agent_resource: str,
    resume_url: str,
    resume_token: str,
    jd_text: str,
    application_id: str,
    job_id: str,
) -> tuple[str, str]:
    """Prep locally, seed the deployed agent session, run it, return (summary, json)."""
    agent_resource = (agent_resource or "").strip()
    resume_url = (resume_url or "").strip()
    if not agent_resource:
        return "Provide the agent resource name.", ""
    if not resume_url:
        return "Provide a resume URL.", ""
    if not (jd_text or "").strip():
        return "Provide a job description.", ""

    app_id = application_id.strip() or str(uuid.uuid4())
    job = job_id.strip() or str(uuid.uuid4())

    # 1. Fetch resume bytes (reuse the app's URL/GCS fetch helper).
    try:
        resume_bytes = ResumeScreeningApp._fetch_url_bytes(
            resume_url, label="resume", auth_token=(resume_token or None)
        )
    except Exception as exc:
        return f"Resume fetch error: {type(exc).__name__}: {exc}", ""
    filename = ResumeScreeningApp._filename_from_url(resume_url) or "resume.pdf"

    # 2. Run the deterministic prep pipeline locally.
    try:
        state = prepare_screening_state(
            application_id=app_id,
            job_id=job,
            resume_bytes=resume_bytes,
            resume_filename=filename,
            jd_text=jd_text.strip(),
        )
    except Exception as exc:
        return f"Prep error: {type(exc).__name__}: {exc}", ""
    state["request_id"] = uuid.uuid4().hex
    state["screening_mode"] = "agent"

    # Crawl the extracted profile URLs LOCALLY via Exa and seed the results into the
    # session. The deployed bare agent does not pre-enrich (auto_enrich_profiles may be
    # off and run_screening_agent_async is bypassed), so without this no links get
    # crawled. fetch_profiles on the agent will skip URLs already enriched here.
    extracted = list(state.get("profile_urls") or [])
    if extracted:
        try:
            results = asyncio.run(enrich_profile_urls_async(state))
            crawled = sum(1 for r in results if isinstance(r, dict) and r.get("ok"))
            enrich_note = (
                f"urls extracted ({len(extracted)}): {extracted}\n"
                f"crawled ok: {crawled}"
            )
        except Exception as exc:  # surface but continue
            enrich_note = (
                f"urls extracted ({len(extracted)}): {extracted}\n"
                f"crawl error: {type(exc).__name__}: {exc}"
            )
    else:
        enrich_note = "urls extracted: 0 (extract_links found no links in this resume)"

    # The brief carries JD/resume/rubric for the LLM to score from.
    brief = build_agent_user_message(state)

    # 3-5. Seed a remote session with the prep state, run the agent, read the result.
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        client = vertexai.Client(project=PROJECT_ID, location=LOCATION)
        remote = client.agent_engines.get(name=agent_resource)

        session = remote.create_session(user_id=app_id, state=state)
        sid = _session_id(session)
        if not sid:
            return f"Could not read session id from: {session!r}", ""

        events = [
            event
            for event in remote.stream_query(user_id=app_id, session_id=sid, message=brief)
        ]

        final = remote.get_session(user_id=app_id, session_id=sid)
        result = _session_state(final).get("screening_result")
    except Exception as exc:
        return f"Agent call error: {type(exc).__name__}: {exc}", ""

    if isinstance(result, dict):
        score = result.get("resume_similarity_score") or {}
        summary = (
            f"{enrich_note}\n"
            f"status: {result.get('resume_screening_status')}\n"
            f"score: {score.get('score') if isinstance(score, dict) else None}\n"
            f"recommendation: {result.get('recommendation')}"
        )
        return summary, json.dumps(result, indent=2)

    return (
        f"{enrich_note}\n"
        "No screening_result in session state (the agent may not have called "
        "submit_screening_result).",
        json.dumps(events, indent=2, default=str),
    )


with gr.Blocks(title="EXAai — prep + deployed agent") as demo:
    gr.Markdown(
        "# EXAai — pipeline prep + deployed agent\n"
        "Runs `prepare_screening_state()` locally, seeds the deployed agent's session "
        "with that state, sends the brief, then reads back `screening_result`. "
        "Auth via Application Default Credentials (`gcloud auth application-default login`)."
    )
    with gr.Row():
        with gr.Column(scale=1):
            agent_resource = gr.Textbox(
                label="Agent resource name",
                value=DEFAULT_AGENT_RESOURCE,
                lines=2,
            )
            resume_url = gr.Textbox(
                label="Resume URL (https:// or gs://)",
                placeholder="https://storage.googleapis.com/bucket/resume.pdf",
            )
            resume_token = gr.Textbox(
                label="Resume Bearer token (optional, for an authenticated https:// URL)",
                type="password",
            )
            jd = gr.Textbox(label="Job description", lines=8, placeholder="Paste the JD here...")
            application_id = gr.Textbox(
                label="application_id (UUID)",
                value="00000000-0000-0000-0000-000000000001",
            )
            job_id = gr.Textbox(
                label="job_id (UUID)",
                value="00000000-0000-0000-0000-000000000002",
            )
            run_btn = gr.Button("Prep + run agent", variant="primary")
        with gr.Column(scale=1):
            summary_out = gr.Textbox(label="Summary", lines=3)
            json_out = gr.Code(label="screening_result / raw events", language="json")

    run_btn.click(
        run_pipeline_then_agent,
        inputs=[agent_resource, resume_url, resume_token, jd, application_id, job_id],
        outputs=[summary_out, json_out],
    )


if __name__ == "__main__":
    demo.launch()
