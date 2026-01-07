"""Helpers for retrieving environment variables and secrets."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

_DOTENV_LOADED = False


def _load_dotenv() -> None:
    global _DOTENV_LOADED  # noqa: PLW0603
    if _DOTENV_LOADED:
        return
    dotenv_path = Path.cwd() / ".env"
    if not dotenv_path.exists():
        return
    try:  # pragma: no cover - optional dependency
        dotenv_module = importlib.import_module("dotenv")
    except ImportError:
        _DOTENV_LOADED = True
        return
    load_dotenv_fn: Callable[..., bool] | None = getattr(dotenv_module, "load_dotenv", None)
    if load_dotenv_fn is None:
        _DOTENV_LOADED = True
        return
    load_dotenv_fn(dotenv_path, override=False)
    _DOTENV_LOADED = True


def get_env(name: str, *, default: str | None = None) -> str | None:
    """Return the value of ``name`` if present and non-empty, else ``default``."""

    _load_dotenv()
    value = os.getenv(name)
    if value is not None and value != "":
        return value
    return default


def require_env(name: str, *, default: str | None = None) -> str:
    """Return the value of ``name`` or raise ``RuntimeError`` if missing."""

    value = get_env(name)
    if value is not None:
        return value
    if default is not None:
        return default
    raise RuntimeError(f"Environment variable '{name}' is required")


def resolve_any(
    names: Sequence[str],
    *,
    default: str | None = None,
    required: bool = False,
) -> str | None:
    """Return the first non-empty value among ``names``.

    Parameters
    ----------
    names:
        Ordered list of environment variable names to query.
    default:
        Fallback value if none of the variables are set.
    required:
        If ``True`` and no value is found, raise ``RuntimeError``.
    """

    for name in names:
        value = get_env(name)
        if value is not None:
            return value
    if default is not None:
        return default
    if required:
        joined = ", ".join(names)
        raise RuntimeError(f"One of the environment variables {joined} must be set")
    return None


def load_env_map(pairs: Iterable[tuple[str, str | None]]) -> dict[str, str]:
    """Return a dict of environment values for the provided ``(name, default)`` pairs."""

    resolved: dict[str, str] = {}
    for name, default in pairs:
        value = get_env(name, default=default)
        if value is not None:
            resolved[name] = value
    return resolved


def ensure_env_loaded() -> None:
    """Ensure the project's ``.env`` file is loaded once."""

    _load_dotenv()


# ruff: noqa: RUF022
__all__ = ["get_env", "require_env", "resolve_any", "load_env_map", "ensure_env_loaded"]
