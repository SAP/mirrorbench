"""Helper filters for common registry metadata queries."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import cast

from mirrorbench.core.registry.entries import RegistryEntry


def filter_by_task(entries: Iterable[RegistryEntry], task: str) -> list[RegistryEntry]:
    matched: list[RegistryEntry] = []
    for entry in entries:
        metadata = entry.metadata
        if metadata is None:
            continue
        supported = cast(Collection[str], getattr(metadata, "supported_tasks", ()))
        if task in supported:
            matched.append(entry)
    return matched


def filter_by_capability(entries: Iterable[RegistryEntry], capability: str) -> list[RegistryEntry]:
    matched: list[RegistryEntry] = []
    for entry in entries:
        metadata = entry.metadata
        if metadata is None:
            continue
        capabilities = cast(Collection[str], getattr(metadata, "capabilities", ()))
        if capability in capabilities:
            matched.append(entry)
    return matched


def filter_metrics_requiring_judge(entries: Iterable[RegistryEntry]) -> list[RegistryEntry]:
    return [entry for entry in entries if getattr(entry.metadata, "needs_judge", False)]


__all__ = [
    "filter_by_capability",
    "filter_by_task",
    "filter_metrics_requiring_judge",
]
