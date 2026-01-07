"""Metadata-aware registry package."""

from __future__ import annotations

from mirrorbench.core.registry.components import (
    BaseDatasetLoader,
    BaseJudge,
    BaseMetric,
    BaseModelClient,
    BaseUserProxyAdapter,
)
from mirrorbench.core.registry.decorators import (
    register_dataset,
    register_judge,
    register_metric,
    register_model_client,
    register_task_driver,
    register_user_proxy,
)
from mirrorbench.core.registry.entries import GROUPS, GroupName, RegistryEntry
from mirrorbench.core.registry.filters import (
    filter_by_capability,
    filter_by_task,
    filter_metrics_requiring_judge,
)
from mirrorbench.core.registry.service import RegistryCatalog

registry = RegistryCatalog()

# ruff: noqa: RUF022
__all__ = [
    "BaseDatasetLoader",
    "BaseJudge",
    "BaseMetric",
    "BaseModelClient",
    "BaseUserProxyAdapter",
    "GROUPS",
    "GroupName",
    "RegistryCatalog",
    "RegistryEntry",
    "filter_by_capability",
    "filter_by_task",
    "filter_metrics_requiring_judge",
    "register_dataset",
    "register_judge",
    "register_metric",
    "register_model_client",
    "register_task_driver",
    "register_user_proxy",
    "registry",
]
