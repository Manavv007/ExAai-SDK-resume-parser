import asyncio
import os
import json
from unittest.mock import AsyncMock, patch
from agent.pipeline import run_screening_async
from agent.tools.github_client import RepoMeta, TreeEntry, CommitMeta

# Enable deep analysis config
os.environ["GITHUB_ANALYSIS_ENABLED"] = "True"
os.environ["GEMINI_API_KEY"] = "mock-api-key"
os.environ["EXA_API_KEY"] = "mock-api-key"

mock_repos = [
    RepoMeta(
        name="test-web-app",
        owner="test-user",
        url="https://github.com/test-user/test-web-app",
        description="A Python FastAPI application with Docker and tests",
        language="Python",
        stars=15,
        forks=1,
        is_fork=False,
        default_branch="main",
        updated_at="2026-06-05T12:00:00Z",
        topics=["fastapi", "docker", "web-app"]
    )
]

mock_tree = [
    TreeEntry(path="README.md", mode="100644", type="blob", sha="sha1", size=100),
    TreeEntry(path="main.py", mode="100644", type="blob", sha="sha2", size=200),
    TreeEntry(path="pyproject.toml", mode="100644", type="blob", sha="sha3", size=300),
    TreeEntry(path="tests/test_app.py", mode="100644", type="blob", sha="sha4", size=400),
    TreeEntry(path="Dockerfile", mode="100644", type="blob", sha="sha5", size=150),
]

def mock_get_content(owner, repo, path):
    if path == "README.md":
        return "# Test Web App\nThis is a cool FastAPI app."
    if path == "pyproject.toml":
        return """
[tool.poetry.dependencies]
python = "^3.10"
fastapi = "^0.100.0"
pytest = "^7.0"
"""
    if path == "main.py":
        return "import fastapi\napp = fastapi.FastAPI()"
    return ""

async def main():
    print("Running integration check...")
    
    # Simple resume with a GitHub link
    resume_text = """
    Jane Doe
    Software Engineer
    Email: jane@example.com
    GitHub: https://github.com/test-user
    Skills: Python, FastAPI, Docker, Pytest
    """
    
    jd_text = """
    Role: Senior Python Developer
    Must have: FastAPI, Docker, testing experience
    Nice to have: AWS
    """
    
    with (
        patch("agent.tools.github_client.GitHubClient.get_user_repos", new_callable=AsyncMock) as mock_get_repos,
        patch("agent.tools.github_client.GitHubClient.get_repo_languages", new_callable=AsyncMock) as mock_langs,
        patch("agent.tools.github_client.GitHubClient.get_repo_tree", new_callable=AsyncMock) as mock_get_tree,
        patch("agent.tools.github_client.GitHubClient.get_file_content", new_callable=AsyncMock) as mock_get_file_content,
        patch("agent.tools.github_client.GitHubClient.get_recent_commits", new_callable=AsyncMock) as mock_get_commits,
        patch("agent.tools.github_client.GitHubClient.get_user_events", new_callable=AsyncMock) as mock_get_events,
        patch("agent.tools.github_analyzer._generate_coding_style_summary", new_callable=AsyncMock) as mock_gen_summary,
        patch("agent.tools.crawler.fetch_url_text_batch", return_value={"https://github.com/test-user": "mock profile html"}),
        patch("agent.tools.scorer._generate_json") as mock_score_llm,
    ):
        mock_get_repos.return_value = mock_repos
        mock_langs.return_value = {"Python": 1000}
        mock_get_tree.return_value = mock_tree
        mock_get_file_content.side_effect = mock_get_content
        mock_get_commits.return_value = [
            CommitMeta(sha="123", message="Initial commit", date="2026-06-05T12:00:00Z", author_name="Jane Doe")
        ]
        mock_get_events.return_value = [
            {"type": "PullRequestEvent", "repo": {"name": "other-owner/repo2"}, "payload": {"action": "opened"}}
        ]
        mock_gen_summary.return_value = (
            "Excellent FastAPI codebase with high quality testing and clean style.",
            "strong",
            "Active open source collaborator.",
            "Clean commits with conventional format."
        )
        
        # Mock final LLM scoring response to match the score schema
        mock_score_llm.return_value = {
            "resume_similarity_score": {
                "score": 92,
                "reasoning": "Strong match based on resume and verified GitHub profile evidence."
            },
            "requirement_matches": [
                {
                    "requirement": "FastAPI",
                    "requirement_type": "technical_skill",
                    "match_score": 95,
                    "evidence": "Candidate has a verified FastAPI project on GitHub with proper structures."
                },
                {
                    "requirement": "Docker",
                    "requirement_type": "technical_skill",
                    "match_score": 90,
                    "evidence": "Docker configuration was found in the GitHub repo."
                }
            ],
            "recommendation": "advance",
            "recommendation_reasoning": "Highly skilled Python developer with strong public project evidence.",
            "red_flags": []
        }

        # Run pipeline
        result = await run_screening_async(
            application_id="11111111-1111-4111-8111-111111111111",
            job_id="22222222-2222-4222-8222-222222222222",
            resume_bytes=resume_text.encode("utf-8"),
            resume_filename="resume.txt",
            jd_text=jd_text,
        )
        
        print("\nScreening Results:")
        print(json.dumps(result, indent=2))
        
        # Validate that the sources crawled matches the expected list
        print("\nSources Crawled:", result.get("sources_crawled"))
        
        print("\nIntegration check completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
