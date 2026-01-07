from __future__ import annotations

from mirrorbench.cache.manager import CacheManager
from mirrorbench.cache.sqlite_backend import SqliteCacheBackend
from mirrorbench.cli import cmd_cache_purge, cmd_cache_stats
from mirrorbench.core.config import CacheConfig
from mirrorbench.core.models.cache import CacheKey
from mirrorbench.io.paths import Paths


class _Args:
    def __init__(self, namespace: str | None = None) -> None:
        self.namespace = namespace


def test_cache_cli_stats_and_purge(tmp_path, capsys, monkeypatch) -> None:
    paths = Paths(tmp_path / "mirrorbench")
    backend = SqliteCacheBackend(paths.cache_db_path())
    config = CacheConfig(enabled=True, ttl_seconds=None)
    manager = CacheManager(backend=backend, config=config)
    manager.set(CacheKey(namespace="demo", key="abc"), b"payload", metadata={"model": "demo"})

    monkeypatch.setattr("mirrorbench.cli.Paths.default", lambda: paths)
    monkeypatch.setattr("mirrorbench.cli.get_cache_manager", lambda *args, **kwargs: manager)

    cmd_cache_stats(_Args())
    captured = capsys.readouterr()
    assert "namespace" in captured.out
    assert "demo" in captured.out

    cmd_cache_purge(_Args(namespace="demo"))
    captured = capsys.readouterr()
    assert "purged 1 entries" in captured.out
