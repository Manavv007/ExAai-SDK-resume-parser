"""GitHub repository analysis tool for the screening agent."""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any

from hermes_tools import terminal


def analyze_github_repo(url: str, tool_context: ToolContext) -> dict[str, Any]:
    """Clone a GitHub repository and analyze its codebase.

    Args:
        url: The GitHub repository URL (e.g., https://github.com/owner/repo)
        tool_context: The tool context for accessing state.

    Returns:
        A dictionary containing analysis results or an error.
    """
    # Validate URL
    if not url.startswith('https://github.com/'):
        return {
            "ok": False,
            "error": "invalid_url",
            "message": "URL must be a GitHub HTTPS URL."
        }

    # Create a temporary directory
    tmpdir = tempfile.mkdtemp(prefix="github_analysis_")
    try:
        # Clone the repository (shallow clone for speed)
        clone_cmd = f"git clone --depth 1 {url} {tmpdir}"
        result = terminal(command=clone_cmd, timeout=60)
        if result.get("exit_code", 0) != 0:
            return {
                "ok": False,
                "error": "clone_failed",
                "message": f"Failed to clone repository: {result.get('output', '')}"
            }

        # Change to the cloned directory
        # We'll run commands in the tmpdir via the workdir parameter
        analysis = {
            "ok": True,
            "repo_url": url,
            "clone_path": tmpdir,
        }

        # Get repository language (by checking file extensions)
        lang_cmd = "find . -type f -name '*.py' -o -name '*.js' -o -name '*.ts' -o -name '*.java' -o -name '*.cpp' -o -name '*.c' -o -name '*.cs' -o -name '*.go' -o -name '*.rs' -o -name '*.rb' -o -name '*.php' | wc -l"
        lang_result = terminal(command=lang_cmd, workdir=tmpdir)
        analysis["file_count_by_extension"] = lang_result.get("output", "0").strip()

        # Get total lines of code (excluding binary files, approximate)
        loc_cmd = "find . -type f -not -path './.git/*' -not -name '*.lock' -not -name '*.png' -not -name '*.jpg' -not -name '*.gif' -not -name '*.pdf' -not -name '*.zip' | xargs wc -l 2>/dev/null | tail -1"
        loc_result = terminal(command=loc_cmd, workdir=tmpdir)
        loc_output = loc_result.get("output", "0").strip()
        # Extract the number (last line of wc -l output)
        lines = loc_output.split()[-1] if loc_output.split() else "0"
        analysis["lines_of_code"] = lines

        # Check for README
        readme_cmd = "find . -maxdepth 2 -type f \\( -iname 'readme*' -o -iname 'readme*.md' -o -iname 'readme*.txt' -o -iname 'readme*.rst' \\) | head -1"
        readme_result = terminal(command=readme_cmd, workdir=tmpdir)
        readme_path = readme_result.get("output", "").strip()
        if readme_path:
            readme_content_cmd = f"head -20 {readme_path}"
            readme_content = terminal(command=readme_content_cmd, workdir=tmpdir)
            analysis["readme_preview"] = readme_content.get("output", "")[:500]
        else:
            analysis["readme_preview"] = ""

        # List top-level directories and files (limit 20)
        list_cmd = "ls -la | head -20"
        list_result = terminal(command=list_cmd, workdir=tmpdir)
        analysis["top_level_listing"] = list_result.get("output", "")

        return analysis

    except Exception as e:
        return {
            "ok": False,
            "error": "unexpected_error",
            "message": str(e)
        }
    finally:
        # Clean up the temporary directory
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)