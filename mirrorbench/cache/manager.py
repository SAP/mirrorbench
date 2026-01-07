"""Cache manager responsible for coordinating backend usage and logging."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from threading import RLock

import structlog

from mirrorbench.cache.sqlite_backend import SqliteCacheBackend
from mirrorbench.core.config import CacheConfig
from mirrorbench.core.models.cache import (
    CacheBackend,
    CacheEntry,
    CacheError,
    CacheKey,
    CacheStats,
)
from mirrorbench.io.paths import Paths

_LOG = structlog.get_logger(__name__)


class CacheManager:
    """High-level cache orchestrator with graceful degradation."""

    def __init__(self, backend: CacheBackend, config: CacheConfig) -> None:
        self._backend = backend
        self._config = config
        self._enabled = bool(config.enabled)
        self._logger = _LOG.bind(backend=config.backend)
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def config(self) -> CacheConfig:
        return self._config

    # ------------------------------------------------------------------
    # Entry helpers
    # ------------------------------------------------------------------
    def get(self, key: CacheKey) -> CacheEntry | None:
        if not self._enabled:
            return None
        try:
            entry = self._backend.get(key)
        except CacheError as exc:  # pragma: no cover - defensive
            self._disable(exc)
            return None
        except Exception as exc:  # pragma: no cover - unexpected
            self._disable(exc)
            return None
        if entry is not None:
            self._logger.info(
                "cache_hit",
                namespace=key.namespace,
                key=key.key,
                version=key.version,
            )
        else:
            self._logger.info(
                "cache_miss",
                namespace=key.namespace,
                key=key.key,
                version=key.version,
            )
        return entry

    def set(
        self,
        key: CacheKey,
        value: bytes,
        *,
        metadata: Mapping[str, str] | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        if not self._enabled:
            return
        expires_at = _compute_expiry(self._config, ttl_seconds)
        meta = {str(k): str(v) for k, v in (metadata or {}).items()}
        entry = CacheEntry(
            value=value,
            stored_at=datetime.now(UTC),
            expires_at=expires_at,
            metadata=meta,
        )
        try:
            self._backend.set(key, entry)
        except CacheError as exc:  # pragma: no cover - defensive
            self._disable(exc)
        except Exception as exc:  # pragma: no cover - unexpected
            self._disable(exc)

    def delete(self, key: CacheKey) -> None:
        if not self._enabled:
            return
        try:
            self._backend.delete(key)
        except CacheError as exc:  # pragma: no cover - defensive
            self._disable(exc)
        except Exception as exc:  # pragma: no cover - unexpected
            self._disable(exc)

    def purge(self, namespace: str | None = None) -> int:
        if not self._enabled:
            return 0
        try:
            return self._backend.purge(namespace)
        except CacheError as exc:  # pragma: no cover - defensive
            self._disable(exc)
            return 0
        except Exception as exc:  # pragma: no cover - unexpected
            self._disable(exc)
            return 0

    def stats(self, namespace: str | None = None) -> list[CacheStats]:
        if not self._enabled:
            return []
        try:
            return self._backend.stats(namespace)
        except CacheError as exc:  # pragma: no cover - defensive
            self._disable(exc)
            return []
        except Exception as exc:  # pragma: no cover - unexpected
            self._disable(exc)
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _disable(self, exc: Exception) -> None:
        with self._lock:
            if not self._enabled:
                return
            self._enabled = False
            self._logger.warning("cache_disabled", reason=str(exc))


_default_manager: CacheManager | None = None
_default_manager_key: tuple[str, str] | None = None
_default_lock = RLock()


def get_cache_manager(paths: Paths, config: CacheConfig) -> CacheManager:
    """Return a process-wide cache manager instance configured for ``paths``."""

    global _default_manager  # noqa: PLW0603
    global _default_manager_key  # noqa: PLW0603

    cache_dir = paths.cache_dir()
    backend_key = (
        str(cache_dir),
        json.dumps(config.model_dump(mode="json"), sort_keys=True),
    )
    with _default_lock:
        if _default_manager is not None and _default_manager_key == backend_key:
            return _default_manager
        backend = _create_backend(paths, config)
        manager = CacheManager(backend=backend, config=config)
        _default_manager = manager
        _default_manager_key = backend_key
        return manager


def _create_backend(paths: Paths, config: CacheConfig) -> CacheBackend:
    if config.backend == "sqlite":
        return SqliteCacheBackend(paths.cache_db_path())
    raise CacheError(f"Unsupported cache backend '{config.backend}'")


def _compute_expiry(config: CacheConfig, ttl_override: int | None) -> datetime | None:
    ttl = ttl_override if ttl_override is not None else config.ttl_seconds
    if ttl is None or ttl <= 0:
        return None
    return datetime.now(UTC) + timedelta(seconds=int(ttl))


__all__ = ["CacheManager", "get_cache_manager"]
