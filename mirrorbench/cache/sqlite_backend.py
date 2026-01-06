"""SQLite backend for cache storage."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from mirrorbench.core.models.cache import (
    CacheBackend,
    CacheEntry,
    CacheKey,
    CacheStats,
)


class SqliteCacheBackend(CacheBackend):
    """SQLite-backed cache backend with TTL support."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._lock = RLock()
        self._conn = self._initialise()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, key: CacheKey) -> CacheEntry | None:
        now = int(time.time())
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT value, stored_at, expires_at, metadata
                FROM cache_entries
                WHERE namespace = ? AND key = ? AND version = ?
                """,
                (key.namespace, key.key, key.version),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            expires_at = row["expires_at"]
            if expires_at is not None and expires_at <= now:
                self._conn.execute(
                    "DELETE FROM cache_entries WHERE namespace = ? AND key = ? AND version = ?",
                    (key.namespace, key.key, key.version),
                )
                self._conn.commit()
                return None
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            stored_at = datetime.fromtimestamp(row["stored_at"], UTC)
            expiry_dt = (
                datetime.fromtimestamp(expires_at, UTC) if isinstance(expires_at, int) else None
            )
            return CacheEntry(
                value=row["value"], stored_at=stored_at, expires_at=expiry_dt, metadata=metadata
            )

    def set(self, key: CacheKey, entry: CacheEntry) -> None:
        stored_at = int(entry.stored_at.timestamp())
        expires_at = int(entry.expires_at.timestamp()) if entry.expires_at else None
        metadata = json.dumps(dict(entry.metadata), sort_keys=True) if entry.metadata else None
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO cache_entries(namespace, key, version, stored_at, expires_at, value, metadata)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key, version)
                DO UPDATE SET stored_at = excluded.stored_at,
                              expires_at = excluded.expires_at,
                              value = excluded.value,
                              metadata = excluded.metadata
                """,
                (
                    key.namespace,
                    key.key,
                    key.version,
                    stored_at,
                    expires_at,
                    entry.value,
                    metadata,
                ),
            )
            self._conn.commit()

    def delete(self, key: CacheKey) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM cache_entries WHERE namespace = ? AND key = ? AND version = ?",
                (key.namespace, key.key, key.version),
            )
            self._conn.commit()

    def purge(self, namespace: str | None = None) -> int:
        now = int(time.time())
        with self._lock:
            self._purge_expired(now)
            if namespace is not None:
                cursor = self._conn.execute(
                    "DELETE FROM cache_entries WHERE namespace = ?",
                    (namespace,),
                )
            else:
                cursor = self._conn.execute("DELETE FROM cache_entries")
            self._conn.commit()
            return cursor.rowcount

    def stats(self, namespace: str | None = None) -> list[CacheStats]:
        now = int(time.time())
        with self._lock:
            self._purge_expired(now)
            if namespace is not None:
                params: tuple[Any, ...] = (namespace,)
                query = """
                    SELECT namespace,
                           COUNT(*) AS entries,
                           COALESCE(SUM(LENGTH(value)), 0) AS total_bytes,
                           MIN(stored_at) AS oldest,
                           MAX(stored_at) AS newest
                    FROM cache_entries
                    WHERE namespace = ?
                    GROUP BY namespace
                    """
            else:
                params = ()
                query = """
                    SELECT namespace,
                           COUNT(*) AS entries,
                           COALESCE(SUM(LENGTH(value)), 0) AS total_bytes,
                           MIN(stored_at) AS oldest,
                           MAX(stored_at) AS newest
                    FROM cache_entries
                    GROUP BY namespace
                    ORDER BY namespace
                    """
            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()
            stats: list[CacheStats] = []
            for row in rows:
                stats.append(
                    CacheStats(
                        namespace=row["namespace"],
                        entries=row["entries"],
                        total_bytes=row["total_bytes"],
                        oldest_entry=_dt_from_epoch(row["oldest"]),
                        newest_entry=_dt_from_epoch(row["newest"]),
                    )
                )
            return stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _initialise(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        with conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=OFF;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    version TEXT NOT NULL,
                    stored_at INTEGER NOT NULL,
                    expires_at INTEGER,
                    value BLOB NOT NULL,
                    metadata TEXT,
                    PRIMARY KEY(namespace, key, version)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS cache_entries_expires_at_idx ON cache_entries(expires_at)"
            )
        return conn

    def _purge_expired(self, now: int) -> None:
        self._conn.execute(
            "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )
        self._conn.commit()


def _dt_from_epoch(value: Any) -> datetime | None:
    if value is None:
        return None
    with suppress(Exception):
        return datetime.fromtimestamp(int(value), UTC)
    return None


__all__ = ["SqliteCacheBackend"]
