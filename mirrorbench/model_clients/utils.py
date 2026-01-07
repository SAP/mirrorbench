"""Shared utilities for model client implementations."""

from __future__ import annotations

from typing import Any


def coerce_int(value: Any) -> int | None:
    """Return ``value`` as ``int`` when possible, otherwise ``None``."""

    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        return None


def coerce_float(value: Any) -> float | None:
    """Return ``value`` as ``float`` when possible, otherwise ``None``."""

    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        return None


def is_chat_client(candidate: Any) -> bool:
    """Return ``True`` when ``candidate`` exposes the chat client surface."""

    if candidate is None:
        return False
    required = ("invoke", "stream")
    for attr in required:
        if not callable(getattr(candidate, attr, None)):
            return False
    return hasattr(candidate, "info")


__all__ = ["coerce_float", "coerce_int", "is_chat_client"]
