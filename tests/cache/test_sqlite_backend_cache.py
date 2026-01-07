from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from mirrorbench.cache.sqlite_backend import SqliteCacheBackend
from mirrorbench.core.models.cache import CacheEntry, CacheKey


def _make_backend(tmp_path: Path) -> SqliteCacheBackend:
    db_path = tmp_path / "cache.db"
    return SqliteCacheBackend(db_path)


def test_set_and_get_roundtrip(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    key = CacheKey(namespace="ns", key="alpha", version="1")
    entry = CacheEntry(value=b"payload")
    backend.set(key, entry)

    result = backend.get(key)

    assert result is not None
    assert result.value == b"payload"
    assert result.metadata == {}


def test_get_removes_expired_entries(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    key = CacheKey(namespace="ns", key="expired", version="1")
    entry = CacheEntry(
        value=b"payload",
        stored_at=datetime.now(UTC) - timedelta(seconds=10),
        expires_at=datetime.now(UTC) - timedelta(seconds=5),
    )
    backend.set(key, entry)

    result = backend.get(key)

    assert result is None
    # entry should be deleted from backing store
    assert backend.get(key) is None


def test_stats_and_purge(tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    key1 = CacheKey(namespace="ns1", key="a", version="1")
    key2 = CacheKey(namespace="ns2", key="b", version="1")
    backend.set(key1, CacheEntry(value=b"a"))
    backend.set(key2, CacheEntry(value=b"b"))

    stats_all = backend.stats()
    assert {s.namespace for s in stats_all} == {"ns1", "ns2"}

    removed = backend.purge(namespace="ns1")
    assert removed == 1
    assert backend.get(key1) is None
    assert backend.get(key2) is not None

    backend.purge()
    assert backend.get(key2) is None
