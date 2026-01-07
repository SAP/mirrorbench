from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mirrorbench.tasks.mirror_conversation  # - ensure mirror driver registration
import mirrorbench.tasks.single_turn  # noqa: F401 - ensure default driver registration
from mirrorbench.core.constants import (
    REGISTRY_GROUP_DATASETS,
    REGISTRY_GROUP_METRICS,
    REGISTRY_GROUP_TASKS,
    REGISTRY_GROUP_USER_PROXIES,
)
from mirrorbench.core.models.errors import PlannerError, RegistryError
from mirrorbench.core.models.plan import (
    DatasetSpec,
    EvalUnit,
    JudgeSpec,
    MetricSpec,
    Plan,
    PlanManifest,
    SkipRecord,
    TaskDriverSpec,
    UserProxySpec,
)
from mirrorbench.core.models.registry import (
    DatasetInfo,
    MetricInfo,
    TaskDriverInfo,
    UserProxyAdapterInfo,
)
from mirrorbench.core.registry import registry
from mirrorbench.core.registry.entries import RegistryEntry

if TYPE_CHECKING:
    from mirrorbench.core.config import JobConfig


PLANNER_VERSION = "0.1.0"


@dataclass(frozen=True)
class _ResolvedProxy:
    spec: UserProxySpec
    entry: RegistryEntry
    adapter_name: str


@dataclass(frozen=True)
class _ResolvedDataset:
    spec: DatasetSpec
    entry: RegistryEntry
    info: DatasetInfo | None


@dataclass(frozen=True)
class _ResolvedMetric:
    spec: MetricSpec
    entry: RegistryEntry


@dataclass(frozen=True)
class _CompatibilityContext:
    adapter: UserProxyAdapterInfo | None
    dataset: DatasetInfo | None
    metric: MetricInfo | None
    task_tag: str | None


class Planner:
    """Resolve job configurations into executable evaluation units."""

    def __init__(self, cfg: JobConfig):
        self._config = cfg
        self._config_hash = _compute_config_hash(cfg)
        self.manifest: PlanManifest | None = None

    @classmethod
    def from_config(cls, cfg: JobConfig | dict[str, Any]) -> Planner:
        from mirrorbench.core.config import JobConfig as _JobConfig

        job_cfg = cfg if isinstance(cfg, _JobConfig) else _JobConfig.model_validate(cfg)
        return cls(job_cfg)

    # ------------------------------------------------------------------
    # Build entry point
    # ------------------------------------------------------------------
    def build(self) -> Plan:
        resolved_proxies = self._resolve_user_proxies()
        resolved_datasets = self._resolve_datasets()
        resolved_metrics = self._resolve_metrics()

        seeds = list(self._config.run.seeds)

        units: list[EvalUnit] = []
        skipped: list[SkipRecord] = []
        metric_judges: dict[str, JudgeSpec] = {}
        task_drivers: dict[str, TaskDriverSpec] = {}

        for metric in resolved_metrics:
            if metric.spec.judge is not None:
                metric_judges[metric.spec.name] = metric.spec.judge

        for dataset in resolved_datasets:
            task_driver_spec = self._resolve_task_driver(dataset)
            task_drivers[dataset.spec.name] = task_driver_spec

        for proxy in resolved_proxies:
            adapter_info = _coerce_metadata(UserProxyAdapterInfo, proxy.entry)
            for dataset in resolved_datasets:
                dataset_info = _coerce_metadata(DatasetInfo, dataset.entry)
                base_context = _CompatibilityContext(
                    adapter=adapter_info,
                    dataset=dataset_info,
                    metric=None,
                    task_tag=_resolve_task(dataset.spec, dataset_info),
                )

                for metric in resolved_metrics:
                    metric_info = _coerce_metadata(MetricInfo, metric.entry)
                    reason = self._compatibility_issue(
                        proxy=proxy,
                        dataset=dataset,
                        metric=metric,
                        context=_CompatibilityContext(
                            adapter=base_context.adapter,
                            dataset=base_context.dataset,
                            metric=metric_info,
                            task_tag=base_context.task_tag,
                        ),
                    )
                    if reason is not None:
                        skipped.append(
                            SkipRecord(
                                proxy=proxy.spec.name,
                                dataset=dataset.spec.name,
                                metric=metric.spec.name,
                                reason=reason,
                            )
                        )
                        continue

                    judge_name = metric.spec.judge.name if metric.spec.judge else None
                    for seed in seeds:
                        units.append(
                            EvalUnit(
                                proxy_name=proxy.spec.name,
                                dataset_name=dataset.spec.name,
                                dataset_split=dataset.spec.split,
                                metric_name=metric.spec.name,
                                seed=int(seed),
                                judge_name=judge_name,
                            )
                        )

        manifest = PlanManifest(
            user_proxies=[item.spec for item in resolved_proxies],
            datasets=[item.spec for item in resolved_datasets],
            metrics=[item.spec for item in resolved_metrics],
            metric_judges=metric_judges,
            task_drivers=task_drivers,
            config_hash=self._config_hash,
            units=units,
            skipped=skipped,
            planner_version=PLANNER_VERSION,
        )

        self.manifest = manifest
        if not units:
            raise PlannerError(
                "Planner produced no executable units. Check skipped combinations for details."
            )
        return manifest.to_plan()

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------
    def _resolve_user_proxies(self) -> list[_ResolvedProxy]:
        resolved: list[_ResolvedProxy] = []
        for spec in self._config.user_proxies:
            adapter_name = spec.adapter or spec.name
            try:
                entry = registry.get(REGISTRY_GROUP_USER_PROXIES, adapter_name)
            except RegistryError as exc:  # pragma: no cover - defensive
                raise PlannerError(f"Unknown user proxy adapter '{adapter_name}'") from exc

            if spec.adapter is None:
                resolved_spec = spec.model_copy(update={"adapter": adapter_name})
            else:
                resolved_spec = spec

            resolved.append(
                _ResolvedProxy(spec=resolved_spec, entry=entry, adapter_name=adapter_name)
            )
        return resolved

    def _resolve_datasets(self) -> list[_ResolvedDataset]:
        resolved: list[_ResolvedDataset] = []
        for _, spec in enumerate(self._config.datasets):
            try:
                entry = registry.get(REGISTRY_GROUP_DATASETS, spec.name)
            except RegistryError as exc:  # pragma: no cover - defensive
                raise PlannerError(f"Unknown dataset '{spec.name}'") from exc
            dataset_info = _coerce_metadata(DatasetInfo, entry)
            resolved.append(_ResolvedDataset(spec=spec, entry=entry, info=dataset_info))
        return resolved

    def _resolve_metrics(self) -> list[_ResolvedMetric]:
        resolved: list[_ResolvedMetric] = []
        for spec in self._config.metrics:
            try:
                entry = registry.get(REGISTRY_GROUP_METRICS, spec.name)
            except RegistryError as exc:  # pragma: no cover - defensive
                raise PlannerError(f"Unknown metric '{spec.name}'") from exc

            metric_info = _coerce_metadata(MetricInfo, entry)
            if metric_info and metric_info.needs_judge and spec.judge is None:
                params = spec.params or {}
                has_inline_client = isinstance(params.get("judge_client_name"), str)
                if not has_inline_client:
                    raise PlannerError(
                        f"Metric '{spec.name}' requires a judge configuration but none was provided."
                    )

            resolved.append(_ResolvedMetric(spec=spec, entry=entry))
        return resolved

    def _resolve_task_driver(self, dataset: _ResolvedDataset) -> TaskDriverSpec:
        override_map = self._config.task_drivers or {}
        override = override_map.get(dataset.spec.name)

        dataset_params = self._extract_dataset_driver_params(dataset.info)
        driver_name = None
        if override is not None:
            if override.driver:
                driver_name = override.driver
            if override.params:
                dataset_params.update(override.params)

        if driver_name is None:
            driver_name = self._dataset_declared_driver(dataset.info)

        if driver_name is None:
            driver_name = self._match_driver_by_task(dataset)

        if driver_name is None:
            raise PlannerError(
                f"Dataset '{dataset.spec.name}' does not declare a task driver and no override was provided"
            )

        return TaskDriverSpec(name=driver_name, params=dataset_params)

    @staticmethod
    def _extract_dataset_driver_params(info: DatasetInfo | None) -> dict[str, Any]:
        if info is None:
            return {}
        extras = getattr(info, "model_extra", {}) or {}
        params = extras.get("task_driver_params")
        if params is None:
            return {}
        if not isinstance(params, Mapping):
            raise PlannerError(
                "Dataset metadata 'task_driver_params' must be a mapping when provided"
            )
        return dict(params)

    @staticmethod
    def _dataset_declared_driver(info: DatasetInfo | None) -> str | None:
        if info is None:
            return None
        extras = getattr(info, "model_extra", {}) or {}
        driver_name = extras.get("task_driver")
        if driver_name is None:
            return None
        if not isinstance(driver_name, str):
            raise PlannerError("Dataset metadata 'task_driver' must be a string when provided")
        return driver_name

    def _match_driver_by_task(self, dataset: _ResolvedDataset) -> str | None:
        task_tag = dataset.spec.task or (dataset.info.supported_tasks if dataset.info else None)
        candidate_tag: str | None = None
        if isinstance(task_tag, str):
            candidate_tag = task_tag
        elif isinstance(task_tag, set) and task_tag:
            candidate_tag = next(iter(sorted(task_tag)))

        if candidate_tag is None:
            return None

        for entry in registry.list_entries(REGISTRY_GROUP_TASKS):
            driver_info = _coerce_metadata(TaskDriverInfo, entry)
            if driver_info and candidate_tag in driver_info.supported_tasks:
                return entry.name
        return None

    # ------------------------------------------------------------------
    # Compatibility checks
    # ------------------------------------------------------------------
    def _compatibility_issue(
        self,
        *,
        proxy: _ResolvedProxy,
        dataset: _ResolvedDataset,
        metric: _ResolvedMetric,
        context: _CompatibilityContext,
    ) -> str | None:
        dataset_info = context.dataset
        if dataset_info and dataset_info.splits and dataset.spec.split not in dataset_info.splits:
            available = ", ".join(sorted(dataset_info.splits))
            return f"dataset split '{dataset.spec.split}' is not available; supported splits: {available}"

        task_tag = context.task_tag
        if task_tag is not None:
            adapter_info = context.adapter
            if (
                adapter_info
                and adapter_info.supported_tasks
                and task_tag not in adapter_info.supported_tasks
            ):
                supported = ", ".join(sorted(adapter_info.supported_tasks))
                return (
                    f"adapter '{proxy.adapter_name}' does not support task '{task_tag}'"
                    f" (supports: {supported})"
                )

            if (
                dataset_info
                and dataset_info.supported_tasks
                and task_tag not in dataset_info.supported_tasks
            ):
                supported = ", ".join(sorted(dataset_info.supported_tasks))
                return (
                    f"dataset '{dataset.spec.name}' does not support task '{task_tag}'"
                    f" (supports: {supported})"
                )

            metric_info = context.metric
            if (
                metric_info
                and metric_info.supported_tasks
                and task_tag not in metric_info.supported_tasks
            ):
                supported = ", ".join(sorted(metric_info.supported_tasks))
                return (
                    f"metric '{metric.spec.name}' does not support task '{task_tag}'"
                    f" (supports: {supported})"
                )

        return None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _compute_config_hash(cfg: JobConfig) -> str:
    payload = cfg.model_dump(mode="json", exclude_none=True)
    normalised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()
    return digest[:16]


def _coerce_metadata(model_cls: type[Any], entry: RegistryEntry) -> Any:
    metadata = entry.metadata
    if metadata is None or not isinstance(metadata, model_cls):
        return None
    return metadata


def _resolve_task(spec: DatasetSpec, info: DatasetInfo | None) -> str | None:
    if spec.task:
        return spec.task
    if info and info.supported_tasks:
        return sorted(info.supported_tasks)[0]
    return None
