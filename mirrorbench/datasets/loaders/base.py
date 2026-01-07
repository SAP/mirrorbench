"""Base contracts for dataset loader backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any


class DatasetLoaderError(RuntimeError):
    """Raised when a loader backend fails to fetch or parse data."""


class DatasetLoaderBackend(ABC):
    """Abstract base class for dataset loader backends."""

    name: str

    def __init__(self, *, params: Mapping[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(params or {})

    @abstractmethod
    def load_split(
        self,
        *,
        split: str,
        limit: int | None = None,
    ) -> Iterable[Mapping[str, Any]]:
        """Yield raw records for the requested split."""

    def shutdown(self) -> None:
        """Optional hook to release loader resources."""

        return None


__all__ = ["DatasetLoaderBackend", "DatasetLoaderError"]
