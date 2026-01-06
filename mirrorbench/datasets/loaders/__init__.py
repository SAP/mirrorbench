"""Loader registry and convenience helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from mirrorbench.datasets.loaders.base import DatasetLoaderBackend, DatasetLoaderError


class LoaderRegistry:
    """Lightweight registry tracking loader backend classes."""

    def __init__(self) -> None:
        self._backends: dict[str, type[DatasetLoaderBackend]] = {}

    def register(self, name: str, backend: type[DatasetLoaderBackend]) -> None:
        key = name.strip()
        if not key:
            raise ValueError("Loader name must be non-empty")
        if key in self._backends:
            raise ValueError(f"Loader '{key}' is already registered")
        backend.name = key
        self._backends[key] = backend

    def get(self, name: str) -> type[DatasetLoaderBackend]:
        try:
            return self._backends[name]
        except KeyError as exc:
            raise DatasetLoaderError(f"Loader '{name}' is not registered") from exc

    def create(self, name: str, *, params: Mapping[str, Any] | None = None) -> DatasetLoaderBackend:
        backend_cls = self.get(name)
        return backend_cls(params=params)

    def names(self) -> set[str]:
        return set(self._backends)


_registry = LoaderRegistry()

T = TypeVar("T", bound=type[DatasetLoaderBackend])


def register_loader(name: str) -> Callable[[T], T]:
    """Class decorator used by loader implementations to self-register."""

    def decorator(cls: T) -> T:
        if not issubclass(cls, DatasetLoaderBackend):  # pragma: no cover - defensive guard
            msg = f"Loader '{name}' must inherit from DatasetLoaderBackend"
            raise TypeError(msg)
        _registry.register(name, cls)
        return cls

    return decorator


def get_loader(name: str) -> type[DatasetLoaderBackend]:
    """Lookup a loader backend class by name."""

    return _registry.get(name)


def create_loader(name: str, *, params: Mapping[str, Any] | None = None) -> DatasetLoaderBackend:
    """Instantiate a loader backend by name."""

    return _registry.create(name, params=params)


def available_loaders() -> set[str]:
    """Return the set of available loader names."""

    return _registry.names()


__all__ = [
    "DatasetLoaderBackend",
    "DatasetLoaderError",
    "available_loaders",
    "create_loader",
    "get_loader",
    "register_loader",
]
