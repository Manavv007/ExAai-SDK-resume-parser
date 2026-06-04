from unittest.mock import MagicMock, patch

from agent.adk_tools import fetch_profile_content, list_candidate_profile_urls


class _FakeState(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def test_list_candidate_profile_urls() -> None:
    ctx = MagicMock()
    ctx.state = _FakeState(
        profile_urls=["https://github.com/janedoe"],
        profile_url_meta=[{"url": "https://github.com/janedoe", "source": "explicit"}],
    )
    result = list_candidate_profile_urls(ctx)
    assert result["count"] == 1
    assert "github.com" in result["urls"][0]


@patch("agent.enrichment.fetch_url_text", return_value="Python projects and OSS contributions.")
def test_fetch_profile_content_success(mock_fetch, test_settings) -> None:
    url = "https://github.com/janedoe"
    ctx = MagicMock()
    ctx.state = _FakeState(profile_urls=[url], enriched_contents=[])

    mock_cache = MagicMock()
    mock_cache.get.return_value = None

    with patch("agent.enrichment.get_url_cache", return_value=mock_cache):
        with patch(
            "agent.enrichment.validate_url",
            return_value=MagicMock(allowed=True, reason=None),
        ):
            with patch(
                "agent.enrichment.check_allowlist",
                return_value=MagicMock(allowed=True, reason=None, domain_category="code"),
            ):
                result = fetch_profile_content(url, ctx)

    assert result["ok"] is True
    assert len(ctx.state["enriched_contents"]) == 1
    mock_fetch.assert_called_once_with(url)


def test_fetch_profile_content_rejects_unknown_url(test_settings) -> None:
    ctx = MagicMock()
    ctx.state = _FakeState(profile_urls=["https://github.com/janedoe"], enriched_contents=[])

    result = fetch_profile_content("https://evil.example.com/x", ctx)
    assert result["ok"] is False
    assert result["error"] == "url_not_in_candidate_list"
