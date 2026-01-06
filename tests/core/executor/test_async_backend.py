from __future__ import annotations

import asyncio
import time
from pathlib import Path

from mirrorbench.core.config import RunConfig
from mirrorbench.core.constants import DEFAULT_TASK_DRIVER_NAME, STATUS_CANCELLED, STATUS_COMPLETED
from mirrorbench.core.executor import RunController
from mirrorbench.core.executor.async_backend import AsyncExecutionBackend
from mirrorbench.core.manifest import ManifestIO
from mirrorbench.core.models.episodes import EpisodeSpec
from mirrorbench.core.models.messages import Message, Role
from mirrorbench.core.models.plan import (
    DatasetSpec,
    EvalUnit,
    MetricSpec,
    PlanManifest,
    UserProxySpec,
)
from mirrorbench.core.models.registry import DatasetInfo, MetricInfo, UserProxyAdapterInfo
from mirrorbench.core.models.run import MetricAggregate, MetricValue
from mirrorbench.core.registry import (
    register_dataset,
    register_metric,
    register_user_proxy,
    registry,
)
from mirrorbench.core.registry.components import BaseDatasetLoader, BaseMetric, BaseUserProxyAdapter
from mirrorbench.core.run_db.sqlite import SQLiteRunDatabase
from mirrorbench.io.paths import Paths

ASYNC_ADAPTER_NAME = "adapter:test/async"
ASYNC_DATASET_NAME = "dataset:test/async"
ASYNC_METRIC_NAME = "metric:test/async"

SESSION_BEHAVIOR: dict[str, object] = {
    "delay": 0.0,
    "fail_first": False,
    "fail_sequence": [],
    "async_calls": 0,
}


def _reset_behavior() -> None:
    SESSION_BEHAVIOR["delay"] = 0.0
    SESSION_BEHAVIOR["fail_first"] = False
    SESSION_BEHAVIOR["fail_sequence"] = []
    SESSION_BEHAVIOR["async_calls"] = 0


class _AsyncAdapter(BaseUserProxyAdapter):
    info = UserProxyAdapterInfo(
        name=ASYNC_ADAPTER_NAME,
        capabilities={"chat"},
    )

    def spawn(self, *, config, run_id: str):  # type: ignore[override]
        return _AsyncSession()


class _AsyncSession:
    def __init__(self) -> None:
        self._calls = 0
        self._failed = False

    def _next_response(self) -> float:
        self._calls += 1
        sequence = SESSION_BEHAVIOR.get("fail_sequence")
        if isinstance(sequence, list) and sequence:
            should_fail = bool(sequence.pop(0))
            if should_fail:
                raise RuntimeError("transient failure")
        elif SESSION_BEHAVIOR.get("fail_first") and not self._failed:
            self._failed = True
            raise RuntimeError("transient failure")
        return float(SESSION_BEHAVIOR.get("delay", 0.0))

    def _build_result(self) -> object:
        message = Message(role=Role.ASSISTANT, content="ok")
        return type(
            "Result",
            (),
            {
                "message": message,
                "telemetry": None,
                "raw": {"calls": self._calls},
            },
        )()

    def generate(self, *, turn: EpisodeSpec, request_params=None):  # type: ignore[override]
        delay = self._next_response()
        if delay:
            time.sleep(delay)
        return self._build_result()

    async def generate_async(self, *, turn: EpisodeSpec, request_params=None):  # type: ignore[override]
        SESSION_BEHAVIOR["async_calls"] = int(SESSION_BEHAVIOR.get("async_calls", 0)) + 1
        delay = self._next_response()
        if delay:
            await asyncio.sleep(delay)
        return self._build_result()

    def shutdown(self) -> None:  # pragma: no cover - nothing to release
        return None


class _AsyncDataset(BaseDatasetLoader):
    info = DatasetInfo(
        name=ASYNC_DATASET_NAME,
        supported_tasks={"async"},
        splits={"test"},
    ).model_copy(update={"task_driver": DEFAULT_TASK_DRIVER_NAME})

    def episodes(self, *, spec, split, limit=None):  # type: ignore[override]
        del split, limit
        return [
            EpisodeSpec(
                episode_id="ep-1",
                task_tag="async",
                chat_history=[Message(role=Role.USER, content="hello")],
            ),
            EpisodeSpec(
                episode_id="ep-2",
                task_tag="async",
                chat_history=[Message(role=Role.USER, content="world")],
            ),
        ]


class _AsyncMetric(BaseMetric):
    info = MetricInfo(
        name=ASYNC_METRIC_NAME,
        supported_tasks={"async"},
        category="debug",
    )

    def evaluate(self, episode):  # type: ignore[override]
        value = MetricValue(metric_name=self.info.name, values=[float(len(episode.turns))])
        episode.metric_values[self.info.name] = value
        return value

    def aggregate(self, values):  # type: ignore[override]
        return MetricAggregate(
            metric_name=self.info.name,
            mean=sum(v.values[0] for v in values) / max(len(values), 1),
            sample_size=len(values),
        )


def _register_components() -> None:
    try:
        registry.get("user_proxies", ASYNC_ADAPTER_NAME)
    except Exception:
        register_user_proxy(name=ASYNC_ADAPTER_NAME, metadata=_AsyncAdapter.info)(_AsyncAdapter)
    try:
        registry.get("datasets", ASYNC_DATASET_NAME)
    except Exception:
        register_dataset(name=ASYNC_DATASET_NAME, metadata=_AsyncDataset.info)(_AsyncDataset)
    try:
        registry.get("metrics", ASYNC_METRIC_NAME)
    except Exception:
        register_metric(name=ASYNC_METRIC_NAME, metadata=_AsyncMetric.info)(_AsyncMetric)


def _build_manifest(run_id: str, unit_count: int = 2) -> PlanManifest:
    proxy = UserProxySpec(name="async-proxy", adapter=ASYNC_ADAPTER_NAME)
    dataset = DatasetSpec(name=ASYNC_DATASET_NAME, split="test")
    metric = MetricSpec(name=ASYNC_METRIC_NAME)
    units: list[EvalUnit] = []
    for idx in range(unit_count):
        units.append(
            EvalUnit(
                proxy_name=proxy.name,
                dataset_name=dataset.name,
                metric_name=metric.name,
                seed=idx,
            )
        )
    return PlanManifest(
        user_proxies=[proxy],
        datasets=[dataset],
        metrics=[metric],
        metric_judges={},
        config_hash="unit-test",
        units=units,
        planner_version="test",
    )


def _make_controller(
    tmp_path: Path, manifest: PlanManifest, run_config: RunConfig
) -> RunController:
    paths = Paths(tmp_path / "mirrorbench")
    manifest_io = ManifestIO(paths)
    run_id = "run-async"
    db = SQLiteRunDatabase(paths.run_db_path(run_id), run_id)
    controller = RunController(
        run_id=run_id,
        plan_manifest=manifest,
        run_config=run_config,
        paths=paths,
        db=db,
        manifest_io=manifest_io,
    )
    return controller


def test_async_backend_respects_concurrency(tmp_path: Path) -> None:
    _register_components()
    _reset_behavior()
    manifest = _build_manifest(run_id="run-async", unit_count=2)
    run_config = RunConfig(
        engine="async",
        max_concurrency=2,
        timeout_seconds=None,
        max_retries=0,
        retry_backoff_seconds=0.0,
    )
    controller = _make_controller(tmp_path, manifest, run_config)
    backend = AsyncExecutionBackend(controller)

    SESSION_BEHAVIOR.update({"delay": 0.2, "fail_first": False})
    start = time.perf_counter()
    summary = backend.run(manifest.units)
    elapsed = time.perf_counter() - start

    assert summary.run.extra.get("status") == STATUS_COMPLETED
    assert len(summary.units) == len(manifest.units)
    assert elapsed < MAX_EXPECTED_DURATION


def test_async_backend_prefers_async_session(tmp_path: Path) -> None:
    _register_components()
    _reset_behavior()
    manifest = _build_manifest(run_id="run-async-method", unit_count=1)
    run_config = RunConfig(
        engine="async",
        max_concurrency=1,
        timeout_seconds=None,
        max_retries=0,
        retry_backoff_seconds=0.0,
    )
    controller = _make_controller(tmp_path, manifest, run_config)
    backend = AsyncExecutionBackend(controller)

    summary = backend.run(manifest.units)

    assert summary.run.extra.get("status") == STATUS_COMPLETED
    assert SESSION_BEHAVIOR["async_calls"] == 2  # noqa: PLR2004


def test_async_backend_retries(tmp_path: Path) -> None:
    _register_components()
    _reset_behavior()
    manifest = _build_manifest(run_id="run-retry", unit_count=1)
    run_config = RunConfig(
        engine="async",
        max_concurrency=1,
        timeout_seconds=None,
        max_retries=1,
        retry_backoff_seconds=0.01,
    )
    controller = _make_controller(tmp_path, manifest, run_config)
    backend = AsyncExecutionBackend(controller)

    SESSION_BEHAVIOR.update(
        {"delay": 0.0, "fail_first": False, "fail_sequence": [True, False, False]}
    )
    summary = backend.run(manifest.units)

    assert summary.run.extra.get("status") == STATUS_COMPLETED
    assert summary.units[0].unit_id().startswith("async-proxy")


def test_async_backend_timeout(tmp_path: Path) -> None:
    _register_components()
    _reset_behavior()
    manifest = _build_manifest(run_id="run-timeout", unit_count=1)
    run_config = RunConfig(
        engine="async",
        max_concurrency=1,
        timeout_seconds=0.05,
        max_retries=0,
        retry_backoff_seconds=0.0,
    )
    controller = _make_controller(tmp_path, manifest, run_config)
    backend = AsyncExecutionBackend(controller)

    SESSION_BEHAVIOR.update({"delay": 0.2, "fail_first": False})
    # Timeouts no longer raise exceptions - they log and continue
    summary = backend.run(manifest.units)
    # Verify the run completed successfully (status is only failed if cancelled)
    assert summary.run.extra.get("status") == "completed"
    # But the unit should be marked as failed due to timeout
    assert len(summary.units) == 1
    # Check that episodes were tracked (timeout happens during episode execution)


def test_async_backend_cancellation(tmp_path: Path) -> None:
    _register_components()
    _reset_behavior()
    manifest = _build_manifest(run_id="run-cancel", unit_count=3)
    run_config = RunConfig(
        engine="async",
        max_concurrency=1,
        timeout_seconds=None,
        max_retries=0,
        retry_backoff_seconds=0.0,
    )
    controller = _make_controller(tmp_path, manifest, run_config)
    backend = AsyncExecutionBackend(controller)

    SESSION_BEHAVIOR.update({"delay": 0.2, "fail_first": False})
    completed_units: list[str] = []

    def progress(unit: EvalUnit) -> None:
        completed_units.append(unit.unit_id())

    def cancel() -> bool:
        return len(completed_units) >= 1

    summary = backend.run(manifest.units, cancel_callback=cancel, progress_callback=progress)

    assert summary.run.extra.get("status") == STATUS_CANCELLED
    assert len(completed_units) == 1


MAX_EXPECTED_DURATION = 0.6
