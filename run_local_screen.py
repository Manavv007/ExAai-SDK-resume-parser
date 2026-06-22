"""Run the screening pipeline LOCALLY through the exact object Agent Engine runs.

This calls ``ResumeScreeningApp.screen(...)`` directly, so the local behavior matches
the deployed Agent Engine behavior (same code path, same inputs). Reads API keys from
your local .env.

Note on auth: there is NO api_key argument here. The standalone FastAPI /screen uses
API_KEYS, but on Agent Platform that is replaced by GCP IAM — the caller authenticates
with Google credentials and is granted permission to invoke the reasoning engine.

Usage:
  python run_local_screen.py path/to/resume.pdf "Senior Python engineer, distributed systems"
  python run_local_screen.py            # defaults to tests/fixtures/sample_resume.pdf
"""

from __future__ import annotations

import base64
import json
import sys
import uuid

from dotenv import load_dotenv

from agent.agent_engine_app import ResumeScreeningApp

load_dotenv()

DEFAULT_RESUME = "tests/fixtures/sample_resume.txt"
DEFAULT_JD = "Senior software engineer with Python and distributed systems experience."


def main() -> None:
    resume_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESUME
    jd_text = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_JD

    with open(resume_path, "rb") as f:
        resume_b64 = base64.b64encode(f.read()).decode()

    # Same lifecycle Agent Engine uses: set_up() once, then screen().
    app = ResumeScreeningApp()
    app.set_up()

    result = app.screen(
        application_id="00000000-0000-0000-0000-000000000001",
        job_id="00000000-0000-0000-0000-000000000002",
        resume_b64=resume_b64,
        resume_filename=resume_path.replace("\\", "/").rsplit("/", 1)[-1],
        jd_text=jd_text,
        request_id=uuid.uuid4().hex,
    )

    print(json.dumps(result, indent=2))
    print("\nstatus:", result.get("resume_screening_status"))
    score = result.get("resume_similarity_score")
    if isinstance(score, dict):
        print("score:", score.get("score"))


if __name__ == "__main__":
    main()
