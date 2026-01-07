"""Cache-related models and protocols."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Identifies a cache entry within a namespace."""

    namespace: str
    key: str
    version: str = "1"
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class CacheEntry:
    """Stored cache payload with optional expiry metadata."""

    value: bytes
    stored_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Aggregate statistics for a cache namespace."""

    namespace: str
    entries: int
    total_bytes: int
    oldest_entry: datetime | None
    newest_entry: datetime | None


class CacheBackend(Protocol):
    """Protocol implemented by cache backends."""

    def get(self, key: CacheKey) -> CacheEntry | None: ...

    def set(self, key: CacheKey, entry: CacheEntry) -> None: ...

    def delete(self, key: CacheKey) -> None: ...

    def purge(self, namespace: str | None = None) -> int: ...

    def stats(self, namespace: str | None = None) -> list[CacheStats]: ...


class CacheError(RuntimeError):
    """Raised when cache operations fail."""


__all__ = ["CacheBackend", "CacheEntry", "CacheError", "CacheKey", "CacheStats"]
