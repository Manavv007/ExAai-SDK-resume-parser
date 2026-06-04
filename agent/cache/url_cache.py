"""TTL URL content cache (SQLite)."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

from agent.config import get_settings


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


class UrlCache:
    def __init__(self, db_path: str | None = None) -> None:
        settings = get_settings()
        self._path = Path(db_path or settings.url_cache_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS url_cache (
                    url_hash TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    content TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def get(self, url: str) -> str | None:
        key = _url_hash(url)
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content, expires_at FROM url_cache WHERE url_hash = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        content, expires_at = row
        if expires_at < now:
            self.delete(url)
            return None
        return content

    def set(self, url: str, content: str, ttl_seconds: int | None = None) -> None:
        settings = get_settings()
        ttl = ttl_seconds if ttl_seconds is not None else settings.cache_ttl_seconds
        expires_at = time.time() + ttl
        key = _url_hash(url)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO url_cache (url_hash, url, content, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(url_hash) DO UPDATE SET
                    content=excluded.content,
                    expires_at=excluded.expires_at
                """,
                (key, url, content, expires_at),
            )
            conn.commit()

    def delete(self, url: str) -> None:
        key = _url_hash(url)
        with self._connect() as conn:
            conn.execute("DELETE FROM url_cache WHERE url_hash = ?", (key,))
            conn.commit()
