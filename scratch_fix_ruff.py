from pathlib import Path
ROOT = Path(".")
p = ROOT / "agent" / "adk_tools.py"
text = p.read_text(encoding="utf-8")
text = text.replace(
    'logger = logging.getLogger("exaai_adk.adk_tools")\n\nfrom google.adk.tools.tool_context',
    "from google.adk.tools.tool_context",
)
anchor = ")\n\n\ndef list_candidate_profile_urls"
if 'logger = logging.getLogger("exaai_adk.adk_tools")' not in text:
    text = text.replace(anchor, ')\n\nlogger = logging.getLogger("exaai_adk.adk_tools")' + anchor, 1)
text = text.replace(
    '"Fix repo_specs before retrying run_sandbox_analysis: call get_github_repo_structures "',
    '"Fix repo_specs before retrying run_sandbox_analysis: "\n                "call get_github_repo_structures "',
    1,
)
p.write_text(text, encoding="utf-8")
print("adk_tools ok")
