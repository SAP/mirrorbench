"""Registration decorators for MirrorBench components."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pydantic import BaseModel

from mirrorbench.core.constants import (
    REGISTRY_GROUP_DATASETS,
    REGISTRY_GROUP_JUDGES,
    REGISTRY_GROUP_METRICS,
    REGISTRY_GROUP_MODEL_CLIENTS,
    REGISTRY_GROUP_TASKS,
    REGISTRY_GROUP_USER_PROXIES,
)
from mirrorbench.core.models.errors import RegistryError
from mirrorbench.core.models.registry import (
    DatasetInfo,
    JudgeInfo,
    MetricInfo,
    ModelClientInfo,
    TaskDriverInfo,
    UserProxyAdapterInfo,
)
from mirrorbench.core.registry.entries import GroupName, RegistryEntry

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from mirrorbench.core.registry.service import RegistryCatalog

F = TypeVar("F", bound=Callable[..., Any])


def _registry() -> RegistryCatalog:
    module = importlib.import_module("mirrorbench.core.registry")
    return cast("RegistryCatalog", module.registry)


def _build_metadata(
    info_cls: type[BaseModel],
    name: str,
    metadata: BaseModel | None,
    info_kwargs: dict[str, Any],
) -> BaseModel | None:
    if metadata is not None and info_kwargs:
        raise RegistryError("Provide either a metadata instance or keyword fields, not both")
    if metadata is not None:
        if hasattr(metadata, "name") and metadata.name != name:
            metadata = metadata.model_copy(update={"name": name})
        return metadata
    if info_cls is BaseModel and not info_kwargs:
        return None
    info_kwargs = {**info_kwargs}
    if "name" not in info_kwargs and hasattr(info_cls, "model_fields"):
        info_kwargs["name"] = name
    return info_cls(**info_kwargs)


def _register(  # noqa: PLR0913
    group: GroupName,
    name: str,
    target: Callable[..., Any],
    metadata: BaseModel | None,
    *,
    version: str | None,
    extras: dict[str, Any] | None,
) -> Callable[..., Any]:
    entry = RegistryEntry(
        group=group,
        name=name,
        factory=target,
        metadata=metadata,
        version=version,
        extras=extras or {},
    )
    _registry().register(entry)
    return target


def register_user_proxy(
    *,
    name: str,
    metadata: UserProxyAdapterInfo | None = None,
    version: str | None = None,
    extras: dict[str, Any] | None = None,
    **info_kwargs: Any,
) -> Callable[[F], F]:
    def decorator(target: F) -> F:
        info = _build_metadata(UserProxyAdapterInfo, name, metadata, info_kwargs)
        _register(REGISTRY_GROUP_USER_PROXIES, name, target, info, version=version, extras=extras)
        return target

    return decorator


def register_dataset(
    *,
    name: str,
    metadata: DatasetInfo | None = None,
    version: str | None = None,
    extras: dict[str, Any] | None = None,
    **info_kwargs: Any,
) -> Callable[[F], F]:
    def decorator(target: F) -> F:
        info = _build_metadata(DatasetInfo, name, metadata, info_kwargs)
        _register(REGISTRY_GROUP_DATASETS, name, target, info, version=version, extras=extras)
        return target

    return decorator


def register_metric(
    *,
    name: str,
    metadata: MetricInfo | None = None,
    version: str | None = None,
    extras: dict[str, Any] | None = None,
    **info_kwargs: Any,
) -> Callable[[F], F]:
    def decorator(target: F) -> F:
        info = _build_metadata(MetricInfo, name, metadata, info_kwargs)
        _register(REGISTRY_GROUP_METRICS, name, target, info, version=version, extras=extras)
        return target

    return decorator


def register_judge(
    *,
    name: str,
    metadata: JudgeInfo | None = None,
    version: str | None = None,
    extras: dict[str, Any] | None = None,
    **info_kwargs: Any,
) -> Callable[[F], F]:
    def decorator(target: F) -> F:
        info = _build_metadata(JudgeInfo, name, metadata, info_kwargs)
        _register(REGISTRY_GROUP_JUDGES, name, target, info, version=version, extras=extras)
        return target

    return decorator


def register_model_client(
    *,
    name: str,
    metadata: ModelClientInfo | None = None,
    version: str | None = None,
    extras: dict[str, Any] | None = None,
    **info_kwargs: Any,
) -> Callable[[F], F]:
    def decorator(target: F) -> F:
        info = _build_metadata(ModelClientInfo, name, metadata, info_kwargs)
        _register(REGISTRY_GROUP_MODEL_CLIENTS, name, target, info, version=version, extras=extras)
        return target

    return decorator


def register_task_driver(
    *,
    name: str,
    metadata: TaskDriverInfo | None = None,
    version: str | None = None,
    extras: dict[str, Any] | None = None,
    **info_kwargs: Any,
) -> Callable[[F], F]:
    def decorator(target: F) -> F:
        info = _build_metadata(TaskDriverInfo, name, metadata, info_kwargs)
        _register(REGISTRY_GROUP_TASKS, name, target, info, version=version, extras=extras)
        return target

    return decorator


__all__ = [
    "register_dataset",
    "register_judge",
    "register_metric",
    "register_model_client",
    "register_task_driver",
    "register_user_proxy",
]
