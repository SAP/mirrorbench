"""Helpers for loading registry contributors."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from mirrorbench.core.models.errors import RegistryError

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from mirrorbench.core.registry.service import RegistryCatalog


def import_namespace(package: str) -> None:
    """Import a module or package so that its decorators run."""

    try:
        import_module(package)
    except ImportError as exc:  # pragma: no cover - import failure path
        raise RegistryError(f"Failed to import namespace '{package}'") from exc


def load_entrypoints(group: str, catalog: RegistryCatalog) -> None:
    """Load registry entrypoints and invoke their registration callbacks."""

    selected = entry_points().select(group=group)
    for ep in selected:
        try:
            hook = ep.load()
        except Exception as exc:  # pragma: no cover - plugin load path
            raise RegistryError(f"Failed to load entrypoint '{ep.name}'") from exc
        hook(catalog)


__all__ = ["import_namespace", "load_entrypoints"]
