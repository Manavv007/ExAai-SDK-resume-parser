from pathlib import Path
ROOT = Path(".")

def sub(path, pairs):
    p = ROOT / path
    t = p.read_text(encoding="utf-8")
    for old, new in pairs:
        if old not in t:
            raise SystemExit(f"missing in {path}: {old[:70]!r}")
        t = t.replace(old, new, 1)
    p.write_text(t, encoding="utf-8")

sub("agent/agent_runner.py", [
    ("4. Turn 3 — submit_screening_result after reading sandbox digest (risk_tier, vulns, secrets, excerpts).",
     "4. Turn 3 — submit_screening_result after reading sandbox digest\n   (risk_tier, vulns, secrets, excerpts)."),
    ("   Penalize aligned-repo material risk: CRITICAL/SEVERE risk_tier means resume_similarity_score should",
     "   Penalize aligned-repo material risk: CRITICAL/SEVERE risk_tier means\n   resume_similarity_score should"),
    ('        "CONTINUATION REQUIRED — your next response MUST be a submit_screening_result tool call only.",',
     '        "CONTINUATION REQUIRED — your next response MUST be a submit_screening_result "\n        "tool call only.",'),
    ('                "3) submit_screening_result with top_file_evaluation for each sandbox top_files path\\n"',
     '                "3) submit_screening_result with top_file_evaluation for each sandbox "\n                "top_files path\\n"'),
    ('                "2) run_sandbox_analysis with focus_paths from get_github_repo_structures\\n"',
     '                "2) run_sandbox_analysis with focus_paths from "\n                "get_github_repo_structures\\n"'),
])

sub("agent/config.py", [
    ('            "When true, the ADK agent orchestrates GitHub structure lookup, sandbox runs, and profile fetches."',
     '            "When true, the ADK agent orchestrates GitHub structure lookup, "\n            "sandbox runs, and profile fetches."'),
])

# read config for exact strings
cfg = (ROOT / "agent/config.py").read_text(encoding="utf-8")
for line in cfg.splitlines():
    if len(line) > 100 and "description" in line.lower() or (len(line)>100 and '="' in line):
        pass
print("agent_runner ok")
