"""Registry entry data structures and helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, cast

from pydantic import BaseModel

from mirrorbench.core.constants import (
    REGISTRY_GROUP_DATASETS,
    REGISTRY_GROUP_JUDGES,
    REGISTRY_GROUP_METRICS,
    REGISTRY_GROUP_MODEL_CLIENTS,
    REGISTRY_GROUP_TASKS,
    REGISTRY_GROUP_USER_PROXIES,
    REGISTRY_GROUPS,
    RegistryGroupName,
)
from mirrorbench.core.models.errors import RegistryError
from mirrorbench.core.models.registry import (
    DatasetInfo,
    JudgeInfo,
    MetricInfo,
    ModelClientInfo,
    UserProxyAdapterInfo,
)

GroupName = RegistryGroupName

GROUPS: tuple[GroupName, ...] = REGISTRY_GROUPS

_METADATA_TYPES: Mapping[GroupName, type[BaseModel]] = {
    REGISTRY_GROUP_USER_PROXIES: UserProxyAdapterInfo,
    REGISTRY_GROUP_DATASETS: DatasetInfo,
    REGISTRY_GROUP_METRICS: MetricInfo,
    REGISTRY_GROUP_MODEL_CLIENTS: ModelClientInfo,
    REGISTRY_GROUP_JUDGES: JudgeInfo,
    REGISTRY_GROUP_TASKS: BaseModel,  # placeholder until task metadata is defined
}


def _split_lazy_import(target: str) -> tuple[str, str]:
    if ":" not in target:
        msg = "lazy_import target must be in 'module:attr' format"
        raise RegistryError(msg)
    module, attr = target.rsplit(":", 1)
    if not module or not attr:
        msg = "lazy_import target must include module and attribute"
        raise RegistryError(msg)
    return module, attr


@dataclass(slots=True)
class RegistryEntry:
    """Entry stored in the registry catalog."""

    group: GroupName
    name: str
    factory: Callable[..., Any] | None = None
    metadata: BaseModel | None = None
    version: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    lazy_import: str | None = None

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        if not self.name:
            raise RegistryError("Registry entry name must not be empty")
        if (self.factory is None) == (self.lazy_import is None):
            raise RegistryError(
                "Registry entry must specify exactly one of 'factory' or 'lazy_import'",
            )
        self._validate_metadata()

    def _validate_metadata(self) -> None:
        if self.metadata is None:
            return
        expected = _METADATA_TYPES.get(self.group)
        if expected is not None and not isinstance(self.metadata, expected):
            raise RegistryError(
                f"Metadata for group '{self.group}' must be an instance of {expected.__name__}",
            )
        if hasattr(self.metadata, "name"):
            metadata_name = self.metadata.name
            if metadata_name and metadata_name != self.name:
                # copy/update to keep metadata immutable (pydantic models are frozen)
                self.metadata = self.metadata.model_copy(update={"name": self.name})

    def resolve_factory(self) -> Callable[..., Any]:
        if self.factory is not None:
            return self.factory
        module_name, attr = _split_lazy_import(self.lazy_import or "")
        try:
            module = import_module(module_name)
        except ImportError as exc:  # pragma: no cover - import resolution error path
            raise RegistryError(
                f"Failed to import '{module_name}' for registry entry '{self.name}'",
            ) from exc
        try:
            candidate = getattr(module, attr)
        except AttributeError as exc:  # pragma: no cover - attribute resolution error path
            raise RegistryError(
                f"Module '{module_name}' has no attribute '{attr}' for entry '{self.name}'",
            ) from exc
        if not callable(candidate):
            raise RegistryError(
                f"Resolved object '{module_name}:{attr}' for entry '{self.name}' is not callable",
            )
        factory = cast(Callable[..., Any], candidate)
        self.factory = factory
        return factory

    def to_dict(self, include_factory: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "group": self.group,
            "name": self.name,
            "version": self.version,
            "extras": dict(self.extras),
        }
        factory_path: str | None = None
        if self.factory is not None:
            factory_path = f"{self.factory.__module__}:{self.factory.__qualname__}"
        if include_factory and factory_path is not None:
            data["factory"] = factory_path
        if self.metadata is not None:
            data["metadata"] = self.metadata.model_dump(mode="json")
        if self.lazy_import is not None:
            data["lazy_import"] = self.lazy_import
        elif factory_path is not None:
            data["lazy_import"] = factory_path
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RegistryEntry:
        data: MutableMapping[str, Any] = dict(payload)
        group = cast(GroupName, data.pop("group"))
        name = cast(str, data.pop("name"))
        metadata_payload = data.pop("metadata", None)
        metadata_model: BaseModel | None = None
        expected = _METADATA_TYPES.get(group)
        if metadata_payload is not None and expected is not None and expected is not BaseModel:
            metadata_model = expected.model_validate(metadata_payload)
        version = cast(str | None, data.pop("version", None))
        extras = dict(cast(dict[str, Any] | None, data.pop("extras", None)) or {})
        lazy_import = cast(str | None, data.pop("lazy_import", None))
        factory_path = cast(str | None, data.pop("factory", None))
        if factory_path and not lazy_import:
            lazy_import = factory_path
        if data:
            raise RegistryError(
                f"Unexpected keys when rebuilding RegistryEntry: {', '.join(data.keys())}"
            )
        return cls(
            group=group,
            name=name,
            factory=None,
            metadata=metadata_model,
            version=version,
            extras=extras,
            lazy_import=lazy_import,
        )


__all__ = ["GROUPS", "GroupName", "RegistryEntry"]
