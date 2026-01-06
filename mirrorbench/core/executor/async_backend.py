"""Asyncio-based execution backend for evaluation units."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import structlog

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
from mirrorbench.core.executor.sync_backend import (
    _aggregate_duration,
    _aggregate_turn_telemetry,
    _call_factory_with_optional_kwargs,
)
from mirrorbench.core.models.errors import TaskDriverError
from mirrorbench.core.models.plan import DatasetSpec, EvalUnit, TaskDriverSpec
from mirrorbench.core.models.run import EpisodeResult, MetricValue, RunSummary
from mirrorbench.core.registry import registry
from mirrorbench.io.artifacts import dump_episode_artifact
from mirrorbench.tasks import TaskDriver

UnitExecutor = Callable[[EvalUnit, RunController], object]
_LOG = structlog.get_logger(__name__)


@dataclass(slots=True)
class _UnitResult:
    unit: EvalUnit
    status: str
    attempts: int
    error: Exception | None = None


class _SemaphoreGuard(AbstractAsyncContextManager[None]):
    def __init__(self, semaphore: asyncio.Semaphore) -> None:
        self._semaphore = semaphore

    async def __aenter__(self) -> None:
        await self._semaphore.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self._semaphore.release()
        return False


@register_backend(name="async", aliases=("asyncio",))
class AsyncExecutionBackend:
    """Execute evaluation units concurrently using asyncio."""

    def __init__(self, controller: RunController) -> None:
        self.controller = controller
        self._logger = _LOG.bind(run_id=controller.run_id)
        self._unit_executor: UnitExecutor | None = None
        self._execution_context: _AsyncExecutionContext | None = None

    def run(
        self,
        units: Iterable[EvalUnit],
        unit_executor: UnitExecutor | None = None,
        *,
        cancel_callback: Callable[[], bool] | None = None,
        progress_callback: Callable[[EvalUnit], None] | None = None,
    ) -> RunSummary:
        unit_list = list(units)
        self.controller.on_run_start()
        self._unit_executor = unit_executor
        self._execution_context = _AsyncExecutionContext(self.controller)
        max_concurrency = max(1, self.controller.run_config.max_concurrency)
        self._logger.info(
            "async_runner_start",
            units=len(unit_list),
            max_concurrency=max_concurrency,
        )
        try:
            summary, errors = asyncio.run(
                self._run_async(
                    unit_list,
                    cancel_callback=cancel_callback,
                    progress_callback=progress_callback,
                )
            )
        finally:
            self._execution_context = None
            self.controller.close()
        # Log errors but don't crash the job - episodes/units are already marked as failed
        if errors:
            self._logger.warning(
                "async_runner_completed_with_errors",
                error_count=len(errors),
                errors=[str(e) for e in errors],
            )
        return summary

    async def _run_async(
        self,
        units: list[EvalUnit],
        *,
        cancel_callback: Callable[[], bool] | None,
        progress_callback: Callable[[EvalUnit], None] | None,
    ) -> tuple[RunSummary, list[Exception]]:
        semaphore = asyncio.Semaphore(max(1, self.controller.run_config.max_concurrency))
        cancel_event = asyncio.Event()
        executed_unit_ids: list[str] = []
        tasks: list[asyncio.Task[object]] = []

        async def worker(unit: EvalUnit) -> _UnitResult:
            async with _SemaphoreGuard(semaphore):
                result = await self._execute_unit(
                    unit,
                    cancel_event=cancel_event,
                    cancel_callback=cancel_callback,
                    progress_callback=progress_callback,
                )
            return result

        for unit in units:
            if cancel_event.is_set():
                break
            if cancel_callback and cancel_callback():
                cancel_event.set()
                break
            tasks.append(asyncio.create_task(worker(unit)))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors: list[Exception] = []
        cancelled = cancel_event.is_set()

        for result in results:
            if isinstance(result, _UnitResult):
                # Track all executed units, regardless of success/failure
                executed_unit_ids.append(result.unit.unit_id())
                if result.status == STATUS_FAILED and result.error is not None:
                    # Log errors but don't fail the entire run
                    errors.append(result.error)
                if result.status == STATUS_CANCELLED:
                    cancelled = True
            elif isinstance(result, asyncio.CancelledError):
                cancelled = True
            elif isinstance(result, Exception):
                errors.append(result)

        self.controller.register_executed_units(executed_unit_ids)

        # Run status is only FAILED if cancelled, not due to unit/episode failures
        status = STATUS_COMPLETED
        if cancelled:
            status = STATUS_CANCELLED

        summary = self.controller.on_run_end(status=status)
        return summary, errors

    async def _execute_unit(
        self,
        unit: EvalUnit,
        *,
        cancel_event: asyncio.Event,
        cancel_callback: Callable[[], bool] | None,
        progress_callback: Callable[[EvalUnit], None] | None,
    ) -> _UnitResult:
        unit_id = unit.unit_id()
        attempts = 0
        last_error: Exception | None = None
        timeout = self.controller.run_config.timeout_seconds
        max_retries = self.controller.run_config.max_retries
        backoff = self.controller.run_config.retry_backoff_seconds
        start_time = time.perf_counter()

        self.controller.on_unit_start(unit)

        while True:
            if cancel_event.is_set():
                telemetry = {
                    "duration_seconds": time.perf_counter() - start_time,
                    "retries": attempts,
                }
                self.controller.on_unit_end(unit, status=STATUS_CANCELLED)
                self.controller.on_unit_telemetry(unit, telemetry)
                self._logger.info("async_unit_cancelled", unit_id=unit_id, attempts=attempts)
                return _UnitResult(unit=unit, status=STATUS_CANCELLED, attempts=attempts)

            if cancel_callback and cancel_callback():
                cancel_event.set()
                continue

            attempts += 1
            try:
                exec_coro = self._run_unit_executor(unit)
                if timeout is not None:
                    await asyncio.wait_for(exec_coro, timeout=timeout)
                else:
                    await exec_coro
                duration = time.perf_counter() - start_time
                telemetry = {
                    "duration_seconds": duration,
                    "retries": attempts - 1,
                }
                self.controller.on_unit_end(unit, status=STATUS_COMPLETED)
                self.controller.on_unit_telemetry(unit, telemetry)
                if progress_callback is not None:
                    progress_callback(unit)
                self._logger.info(
                    "async_unit_completed",
                    unit_id=unit_id,
                    attempts=attempts,
                    duration_seconds=telemetry["duration_seconds"],
                )
                return _UnitResult(unit=unit, status=STATUS_COMPLETED, attempts=attempts)
            except TimeoutError:
                last_error = TimeoutError(
                    f"Evaluation unit {unit_id} timed out after {timeout} seconds"
                )
                self._logger.warning("async_unit_timeout", unit_id=unit_id, attempts=attempts)
            except Exception as exc:  # pragma: no cover - safety net for unknown errors
                last_error = exc
                self._logger.warning(
                    "async_unit_exception", unit_id=unit_id, attempts=attempts, error=str(exc)
                )

            if attempts > max_retries:
                duration = time.perf_counter() - start_time
                telemetry = {
                    "duration_seconds": duration,
                    "retries": attempts - 1,
                }
                self.controller.on_unit_end(unit, status=STATUS_FAILED)
                self.controller.on_unit_telemetry(unit, telemetry)
                if last_error is None:
                    last_error = RuntimeError(f"Evaluation unit {unit_id} failed")
                return _UnitResult(
                    unit=unit, status=STATUS_FAILED, attempts=attempts, error=last_error
                )

            delay = backoff * attempts if backoff else 0.0
            if delay:
                await asyncio.sleep(delay)

    async def _run_unit_executor(self, unit: EvalUnit) -> None:
        if self._unit_executor is None:
            assert self._execution_context is not None
            await self._execution_context.execute(unit)
            return
        if inspect.iscoroutinefunction(self._unit_executor):
            await self._unit_executor(unit, self.controller)
            return
        result = await asyncio.to_thread(self._unit_executor, unit, self.controller)
        if inspect.isawaitable(result):
            await result


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


class _AsyncExecutionContext:
    def __init__(self, controller: RunController) -> None:
        self.controller = controller
        manifest = controller.plan_manifest
        self._proxy_specs = {spec.name: spec for spec in manifest.user_proxies}
        self._dataset_specs = {(spec.name, spec.split): spec for spec in manifest.datasets}
        self._metric_specs = {spec.name: spec for spec in manifest.metrics}
        self._task_driver_specs = manifest.task_drivers

    async def execute(self, unit: EvalUnit) -> None:
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
                result = await self._process_episode(
                    unit=unit,
                    episode_spec=episode_spec,
                    components=components,
                )
                if result is not None:
                    episodes_successful += 1
                    metric_value, telemetry_summary = result
                    metric_values.append(metric_value)
                    for key, value in telemetry_summary.items():
                        telemetry_totals[key] += value
                else:
                    episodes_failed += 1

            telemetry_totals["episodes_successful"] = float(episodes_successful)
            telemetry_totals["episodes_failed"] = float(episodes_failed)
            telemetry_totals["episodes_total"] = float(episodes_successful + episodes_failed)

            self._record_aggregates(unit, components.metric, metric_values, telemetry_totals)
        finally:
            await asyncio.to_thread(components.shutdown)

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

    async def _process_episode(
        self,
        *,
        unit: EvalUnit,
        episode_spec: Any,
        components: _UnitComponents,
    ) -> tuple[MetricValue, dict[str, float]] | None:
        self.controller.on_episode_start(unit, episode_spec.episode_id)
        try:
            execution = await components.driver.run_episode_async(
                episode=episode_spec,
                proxy_session=components.session,
                run_id=self.controller.run_id,
            )
            artifact = execution.artifact
            metric_value = await asyncio.to_thread(components.metric.evaluate, artifact)
            artifact.metric_values[metric_value.metric_name] = metric_value

            artifact_path = await asyncio.to_thread(
                dump_episode_artifact,
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
            self.controller._counter_failed.add(1)
            _LOG.warning(
                "episode_failed",
                unit_id=unit.unit_id(),
                episode_id=episode_spec.episode_id,
                error=str(exc),
            )
            return None
        else:
            self.controller.on_episode_end(unit, episode_result)
            return metric_value, telemetry_summary

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
                dataset_spec = candidates[0]
        return dataset_spec
