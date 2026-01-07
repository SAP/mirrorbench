"""Registry utilities for execution backends."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mirrorbench.core.executor.controller import RunController

_BackendFactory = Callable[["RunController"], object]
_BACKENDS: dict[str, _BackendFactory] = {}
_ALIASES: dict[str, str] = {}


def register_backend(*, name: str, aliases: Sequence[str] | None = None) -> Callable[[type], type]:
    """Register a backend class with optional aliases."""

    def decorator(cls: type) -> type:
        normalized = name.strip().lower()
        if normalized in _BACKENDS:
            raise ValueError(f"Backend '{name}' already registered")

        def factory(controller: RunController) -> object:
            return cls(controller)

        _BACKENDS[normalized] = factory
        for alias in aliases or ():
            alias_normalized = alias.strip().lower()
            if alias_normalized in _ALIASES:
                raise ValueError(f"Backend alias '{alias}' already registered")
            if alias_normalized in _BACKENDS:
                raise ValueError(
                    f"Backend alias '{alias}' conflicts with registered backend '{alias_normalized}'",
                )
            _ALIASES[alias_normalized] = normalized
        return cls

    return decorator


def resolve_backend(name: str) -> _BackendFactory:
    """Return the factory associated with ``name`` or raise ``KeyError``."""

    normalized_input = name.strip().lower()
    normalized = _ALIASES.get(normalized_input, normalized_input)
    factory = _BACKENDS.get(normalized)
    if factory is None:
        raise KeyError(f'No backend registered under name or alias "{normalized}"')
    return factory
