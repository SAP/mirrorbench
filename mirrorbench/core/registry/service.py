"""Registry catalog implementation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from threading import RLock
from typing import Any

from mirrorbench.core.models.errors import RegistryError
from mirrorbench.core.registry.entries import GROUPS, GroupName, RegistryEntry
from mirrorbench.core.registry.loader import import_namespace, load_entrypoints


class RegistryCatalog:
    """In-memory catalog that stores registry entries grouped by type."""

    def __init__(self) -> None:
        self._entries: dict[GroupName, dict[str, RegistryEntry]] = {group: {} for group in GROUPS}
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, entry: RegistryEntry) -> None:
        with self._lock:
            table = self._entries[entry.group]
            if entry.name in table:
                raise RegistryError(
                    f"Registry entry '{entry.name}' already registered in group '{entry.group}'",
                )
            table[entry.name] = entry

    def register_many(self, entries: Iterable[RegistryEntry]) -> None:
        for entry in entries:
            self.register(entry)

    def clear(self) -> None:
        with self._lock:
            for table in self._entries.values():
                table.clear()

    def register_factory(  # noqa: PLR0913
        self,
        group: GroupName,
        name: str,
        factory: Callable[..., Any],
        *,
        metadata: Any | None = None,
        version: str | None = None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        entry = RegistryEntry(
            group=group,
            name=name,
            factory=factory,
            metadata=metadata,
            version=version,
            extras=extras or {},
        )
        self.register(entry)

    def register_lazy(  # noqa: PLR0913
        self,
        group: GroupName,
        name: str,
        lazy_import: str,
        *,
        metadata: Any | None = None,
        version: str | None = None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        entry = RegistryEntry(
            group=group,
            name=name,
            factory=None,
            metadata=metadata,
            version=version,
            extras=extras or {},
            lazy_import=lazy_import,
        )
        self.register(entry)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def get(self, group: GroupName, name: str) -> RegistryEntry:
        table = self._entries.get(group)
        if table is None:
            raise RegistryError(f"Unknown registry group '{group}'")
        try:
            return table[name]
        except KeyError as exc:
            raise RegistryError(f"Registry entry '{name}' not found in group '{group}'") from exc

    def factory(self, group: GroupName, name: str, *, load: bool = True) -> Any:
        entry = self.get(group, name)
        if not load:
            return entry.factory if entry.factory is not None else entry.lazy_import
        return entry.resolve_factory()

    def list_entries(
        self,
        group: GroupName,
        *,
        include_metadata: bool = True,
    ) -> list[RegistryEntry]:
        table = self._entries.get(group)
        if table is None:
            raise RegistryError(f"Unknown registry group '{group}'")
        entries = list(table.values())
        entries.sort(key=lambda e: e.name)
        if include_metadata:
            return entries
        return entries

    def find(
        self,
        group: GroupName,
        predicate: Callable[[RegistryEntry], bool],
    ) -> list[RegistryEntry]:
        if not callable(predicate):
            raise RegistryError("Registry find predicate must be callable")
        return [entry for entry in self.list_entries(group) if predicate(entry)]

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {
            group: [entry.to_dict(include_factory=False) for entry in self.list_entries(group)]
            for group in GROUPS
        }

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def load_namespace(self, package: str) -> None:
        import_namespace(package)

    def load_entrypoints(self, group: str = "mirrorbench.plugins") -> None:
        load_entrypoints(group, self)


__all__ = ["RegistryCatalog"]
