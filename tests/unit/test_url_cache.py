from agent.cache.url_cache import UrlCache


def test_url_cache_set_get_and_expiry(tmp_path, test_settings) -> None:
    db = tmp_path / "cache.db"
    cache = UrlCache(str(db))
    cache.set("https://github.com/example", "content", ttl_seconds=3600)
    assert cache.get("https://github.com/example") == "content"

    cache.set("https://github.com/old", "old", ttl_seconds=-10)
    assert cache.get("https://github.com/old") is None
