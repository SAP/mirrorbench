"""Synchronous execution backend."""

from __future__ import annotations

import contextlib
import inspect
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import structlog

import mirrorbench.tasks.single_turn  # noqa: F401 - ensure default driver registration
from mirrorbench.core.constants import (
    DEFAULT_TASK_DRIVER_NAME,
    REGISTRY_GROUP_DATASETS,
    REGISTRY_GROUP_METRICS,
    REGISTRY_GROUP_TASKS,
    REGISTRY_GROUP_USER_PROXIES,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
)
from mirrorbench.core.executor.backend_registry import register_backend
from mirrorbench.core.executor.controller import RunController
from mirrorbench.core.models.errors import TaskDriverError
from mirrorbench.core.models.messages import TurnTelemetry
from mirrorbench.core.models.plan import DatasetSpec, EvalUnit, TaskDriverSpec
from mirrorbench.core.models.run import EpisodeResult, MetricValue, RunSummary
from mirrorbench.core.registry import registry
from mirrorbench.io.artifacts import dump_episode_artifact
from mirrorbench.tasks import TaskDriver

UnitExecutor = Callable[[EvalUnit, RunController], None]


@register_backend(name="sync", aliases=("synchronous",))
class SyncExecutionBackend:
    """Run evaluation units sequentially using the provided controller."""

    def __init__(self, controller: RunController) -> None:
        self.controller = controller

    def run(
        self,
        units: Iterable[EvalUnit],
        unit_executor: UnitExecutor | None = None,
        *,
        cancel_callback: Callable[[], bool] | None = None,
        progress_callback: Callable[[EvalUnit], None] | None = None,
    ) -> RunSummary:
        """Execute evaluation units sequentially."""

        executor = unit_executor or _default_unit_executor(self.controller)
        execution_status = STATUS_COMPLETED
        summary: RunSummary | None = None
        executed_unit_ids: list[str] = []
        errors: list[Exception] = []

        self.controller.on_run_start()
        for unit in units:
            if cancel_callback and cancel_callback():
                execution_status = STATUS_CANCELLED
                break

            self.controller.on_unit_start(unit)
            unit_status = STATUS_COMPLETED
            try:
                executor(unit, self.controller)
            except Exception as exc:  # Unit-level failure (setup/initialization error)
                unit_status = STATUS_FAILED
                # Note: execution_status remains COMPLETED unless cancelled
                # Failed units are logged but don't fail the entire run
                errors.append(exc)
                logger = structlog.get_logger(__name__)
                logger.warning(
                    "unit_failed",
                    unit_id=unit.unit_id(),
                    error=str(exc),
                )
            finally:
                self.controller.on_unit_end(unit, status=unit_status)
                # Track all units that started execution (successful or failed)
                executed_unit_ids.append(unit.unit_id())
                if progress_callback is not None:
                    progress_callback(unit)

            if cancel_callback and cancel_callback():
                execution_status = STATUS_CANCELLED
                break

        self.controller.register_executed_units(executed_unit_ids)
        summary = self.controller.on_run_end(status=execution_status)
        self.controller.close()

        # Log errors but don't crash - units and episodes are already marked as failed
        if errors:
            logger = structlog.get_logger(__name__)
            logger.warning(
                "sync_runner_completed_with_errors",
                error_count=len(errors),
                errors=[str(e) for e in errors],
            )

        return summary


def _default_unit_executor(controller: RunController) -> UnitExecutor:
    return _ExecutionContext(controller).execute


@dataclass(slots=True)
class _UnitComponents:
    adapter: Any
    dataset: Any
    metric: Any
    session: Any
    driver: TaskDriver

    def shutdown(self) -> None:
        for component in (self.session, self.adapter, self.dataset, self.metric, self.driver):
            with contextlib.suppress(AttributeError):
                component.shutdown()


class _ExecutionContext:
    """Encapsulate default unit execution orchestration."""

    def __init__(self, controller: RunController) -> None:
        self.controller = controller
        manifest = controller.plan_manifest
        self._proxy_specs = {spec.name: spec for spec in manifest.user_proxies}
        # Use composite key (name, split) to support multiple splits of same dataset
        self._dataset_specs = {(spec.name, spec.split): spec for spec in manifest.datasets}
        self._metric_specs = {spec.name: spec for spec in manifest.metrics}
        self._task_driver_specs = manifest.task_drivers

    def execute(self, unit: EvalUnit, _: RunController) -> None:
        components = self._instantiate_components(unit)
        metric_values: list[MetricValue] = []
        telemetry_totals: defaultdict[str, float] = defaultdict(float)
        episodes_successful = 0
        episodes_failed = 0

        try:
            dataset_spec = self._resolve_dataset_spec(unit)
            if dataset_spec is None:  # pragma: no cover - defensive
                raise RuntimeError(
                    f"Plan manifest is missing specifications for unit {unit.unit_id()}"
                )
            for episode_spec in components.dataset.episodes(
                spec=dataset_spec, split=dataset_spec.split
            ):
                result = self._process_episode(
                    unit=unit,
                    episode_spec=episode_spec,
                    components=components,
                )
                # Track episode success/failure and only aggregate successful episodes
                if result is not None:
                    episodes_successful += 1
                    metric_value, telemetry_summary = result
                    metric_values.append(metric_value)
                    for key, value in telemetry_summary.items():
                        telemetry_totals[key] += value
                else:
                    episodes_failed += 1

            # Add episode counts to telemetry
            telemetry_totals["episodes_successful"] = float(episodes_successful)
            telemetry_totals["episodes_failed"] = float(episodes_failed)
            telemetry_totals["episodes_total"] = float(episodes_successful + episodes_failed)

            self._record_aggregates(unit, components.metric, metric_values, telemetry_totals)
        finally:
            components.shutdown()

    # ------------------------------------------------------------------
    # Component helpers
    # ------------------------------------------------------------------
    def _instantiate_components(self, unit: EvalUnit) -> _UnitComponents:
        proxy_spec = self._proxy_specs.get(unit.proxy_name)
        dataset_spec = self._resolve_dataset_spec(unit)
        metric_spec = self._metric_specs.get(unit.metric_name)

        if proxy_spec is None or dataset_spec is None or metric_spec is None:  # pragma: no cover
            raise RuntimeError(f"Plan manifest is missing specifications for unit {unit.unit_id()}")

        adapter_name = proxy_spec.adapter
        if adapter_name is None:  # pragma: no cover - defensive
            raise RuntimeError(f"Proxy '{proxy_spec.name}' is missing an adapter name")

        adapter_factory = registry.factory(REGISTRY_GROUP_USER_PROXIES, adapter_name)
        adapter = adapter_factory()
        configure_cache = getattr(adapter, "configure_cache", None)
        if callable(configure_cache):
            configure_cache(
                cache_config=self.controller.run_config.cache,
                paths=self.controller.paths,
            )

        dataset_factory = registry.factory(REGISTRY_GROUP_DATASETS, dataset_spec.name)
        dataset = _call_factory_with_optional_kwargs(dataset_factory, paths=self.controller.paths)

        metric_factory = registry.factory(REGISTRY_GROUP_METRICS, metric_spec.name)
        metric_params = dict(metric_spec.params or {})
        metrics_config = self.controller.run_config.metrics
        if "bootstrap" not in metric_params and metrics_config.bootstrap is not None:
            metric_params["bootstrap"] = metrics_config.bootstrap.model_dump()
        metric = metric_factory(**metric_params)
        configure_metric_cache = getattr(metric, "configure_cache", None)
        if callable(configure_metric_cache):
            configure_metric_cache(
                cache_config=self.controller.run_config.cache,
                paths=self.controller.paths,
            )

        session = adapter.spawn(config=proxy_spec, run_id=self.controller.run_id)

        driver_spec = self._task_driver_specs.get(dataset_spec.name)
        if driver_spec is None:
            driver_spec = TaskDriverSpec(name=DEFAULT_TASK_DRIVER_NAME, params={})
        driver_factory = registry.factory(REGISTRY_GROUP_TASKS, driver_spec.name)
        driver: TaskDriver = driver_factory()
        driver_params = dict(driver_spec.params or {})
        try:
            driver.setup(
                run_id=self.controller.run_id,
                run_config=self.controller.run_config,
                paths=self.controller.paths,
                dataset=dataset_spec,
                params=driver_params,
            )
        except TaskDriverError as exc:
            raise RuntimeError(
                f"Failed to initialise task driver '{driver_spec.name}' for dataset '{dataset_spec.name}'"
            ) from exc

        return _UnitComponents(
            adapter=adapter,
            dataset=dataset,
            metric=metric,
            session=session,
            driver=driver,
        )

    # ------------------------------------------------------------------
    # Episode processing
    # ------------------------------------------------------------------
    def _process_episode(
        self,
        *,
        unit: EvalUnit,
        episode_spec: Any,
        components: _UnitComponents,
    ) -> tuple[MetricValue, dict[str, float]] | None:
        self.controller.on_episode_start(unit, episode_spec.episode_id)
        try:
            execution = components.driver.run_episode(
                episode=episode_spec,
                proxy_session=components.session,
                run_id=self.controller.run_id,
            )
            artifact = execution.artifact
            metric_value = components.metric.evaluate(artifact)
            artifact.metric_values[metric_value.metric_name] = metric_value

            artifact_path = dump_episode_artifact(
                paths=self.controller.paths,
                run_id=self.controller.run_id,
                unit=unit,
                artifact=artifact,
            )

            telemetry_summary = _aggregate_turn_telemetry(execution.turn_telemetries)
            duration = _aggregate_duration(execution.turn_telemetries)

            episode_result = EpisodeResult(
                unit=unit,
                episode_id=episode_spec.episode_id,
                status=STATUS_COMPLETED,
                metric_values={metric_value.metric_name: metric_value},
                telemetry_summary=telemetry_summary,
                duration_s=duration,
                artifact_path=artifact_path,
            )
        except Exception as exc:  # Log and skip failed episodes instead of crashing
            failed_result = EpisodeResult(
                unit=unit,
                episode_id=episode_spec.episode_id,
                status=STATUS_FAILED,
                summary=str(exc),
            )
            self.controller.on_episode_end(unit, failed_result)
            # Increment failed counter in controller
            self.controller._counter_failed.add(1)
            # Log the failure but don't crash - return None to indicate failure
            logger = structlog.get_logger(__name__)
            logger.warning(
                "episode_failed",
                unit_id=unit.unit_id(),
                episode_id=episode_spec.episode_id,
                error=str(exc),
            )
            return None
        else:
            self.controller.on_episode_end(unit, episode_result)
            return metric_value, telemetry_summary

    # ------------------------------------------------------------------
    # Finalization helpers
    # ------------------------------------------------------------------
    def _record_aggregates(
        self,
        unit: EvalUnit,
        metric: Any,
        metric_values: list[MetricValue],
        telemetry_totals: dict[str, float],
    ) -> None:
        if metric_values:
            aggregate = metric.aggregate(metric_values)
            self.controller.on_metric_aggregate(unit, aggregate)

        if telemetry_totals:
            self.controller.on_unit_telemetry(unit, dict(telemetry_totals))

    def _resolve_dataset_spec(self, unit: EvalUnit) -> DatasetSpec | None:
        dataset_spec = self._dataset_specs.get((unit.dataset_name, unit.dataset_split))
        if dataset_spec is None:
            candidates = [
                spec for (name, _), spec in self._dataset_specs.items() if name == unit.dataset_name
            ]
            if len(candidates) == 1:
                # Fall back to the only available split for this dataset when the unit did not
                # specify one (keeps legacy manifests working without blowing up execution).
                dataset_spec = candidates[0]
        return dataset_spec


def _call_factory_with_optional_kwargs(factory: Callable[..., Any], **kwargs: Any) -> Any:
    """Instantiate ``factory`` while supplying supported keyword arguments."""

    if not kwargs:
        return factory()

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):  # pragma: no cover - builtins or C-extensions
        signature = None

    if signature is None:
        return factory()

    parameters = signature.parameters
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )
    provided: dict[str, Any] = {}
    for name, value in kwargs.items():
        if name in parameters or accepts_var_kwargs:
            provided[name] = value

    return factory(**provided) if provided else factory()


def _aggregate_turn_telemetry(turns: list[TurnTelemetry]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for telemetry in turns:
        partial = _summarize_single_turn(telemetry)
        for key, value in partial.items():
            summary[key] = summary.get(key, 0.0) + value
    return summary


def _aggregate_duration(turns: list[TurnTelemetry]) -> float | None:
    total = 0.0
    found = False
    for telemetry in turns:
        value = telemetry.total_response_time
        if value is not None:
            total += float(value)
            found = True
    return total if found else None


def _summarize_single_turn(telemetry: TurnTelemetry | None) -> dict[str, float]:
    if telemetry is None:
        return {}

    numeric_fields = {
        "tokens_input": telemetry.tokens_input,
        "tokens_output": telemetry.tokens_output,
        "total_response_time": telemetry.total_response_time,
        "time_to_first_token": telemetry.time_to_first_token,
        "time_per_output_token": telemetry.time_per_output_token,
        "cost_usd": telemetry.cost_usd,
    }

    summary: dict[str, float] = {}
    for key, value in numeric_fields.items():
        coerced = _safe_float(value)
        if coerced is not None:
            summary[key] = coerced
    return summary


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
