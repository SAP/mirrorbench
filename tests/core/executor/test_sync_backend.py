from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mirrorbench.core.config import RunConfig, ScorecardConfig
from mirrorbench.core.constants import (
    DEFAULT_TASK_DRIVER_NAME,
    REGISTRY_GROUP_DATASETS,
    REGISTRY_GROUP_METRICS,
    REGISTRY_GROUP_USER_PROXIES,
    STATUS_COMPLETED,
)
from mirrorbench.core.executor import RunController, SyncExecutionBackend
from mirrorbench.core.manifest import ManifestIO
from mirrorbench.core.models.episodes import EpisodeArtifact, EpisodeSpec
from mirrorbench.core.models.errors import RegistryError
from mirrorbench.core.models.messages import Message, Role, TurnTelemetry
from mirrorbench.core.models.plan import (
    DatasetSpec,
    EvalUnit,
    MetricSpec,
    PlanManifest,
    UserProxySpec,
)
from mirrorbench.core.models.registry import DatasetInfo, MetricInfo, UserProxyAdapterInfo
from mirrorbench.core.models.run import EpisodeResult, MetricAggregate, MetricValue
from mirrorbench.core.registry import (
    BaseDatasetLoader,
    BaseMetric,
    BaseUserProxyAdapter,
    register_dataset,
    register_metric,
    register_user_proxy,
    registry,
)
from mirrorbench.core.run_db.sqlite import SQLiteRunDatabase
from mirrorbench.io.paths import Paths

ADAPTER_NAME = "adapter:test/echo"
DATASET_NAME = "dataset:test/static"
METRIC_NAME = "metric:test/length"


@dataclass(slots=True)
class _StubGenerationResult:
    message: Message
    telemetry: TurnTelemetry | None = None


class _EchoAdapter(BaseUserProxyAdapter):
    info = UserProxyAdapterInfo(
        name=ADAPTER_NAME,
        capabilities={"chat"},
        supported_tasks={"echo"},
    )

    def spawn(self, *, config: UserProxySpec, run_id: str) -> _EchoSession:
        return _EchoSession(config.name)


class _EchoSession:
    def __init__(self, proxy_name: str) -> None:
        self._proxy_name = proxy_name

    def generate(
        self, *, turn: EpisodeSpec, request_params: dict[str, Any] | None = None
    ) -> _StubGenerationResult:
        message = Message(role=Role.USER, content=f"{self._proxy_name}:{turn.episode_id}")
        telemetry = TurnTelemetry(tokens_input=1, tokens_output=1, total_response_time=0.05)
        return _StubGenerationResult(message=message, telemetry=telemetry)

    def shutdown(self) -> None:  # pragma: no cover - nothing to release for stub
        return None


class _StaticDataset(BaseDatasetLoader):
    info = DatasetInfo(
        name=DATASET_NAME,
        supported_tasks={"echo"},
        splits={"test"},
    ).model_copy(update={"task_driver": DEFAULT_TASK_DRIVER_NAME})

    def episodes(
        self,
        *,
        spec: DatasetSpec,
        split: str,
        limit: int | None = None,
    ) -> list[EpisodeSpec]:  # type: ignore[override]
        del spec  # unused in stub implementation
        if split != "test":  # pragma: no cover - defensive
            return []
        turns = [
            EpisodeSpec(
                episode_id="ep-1",
                task_tag="echo",
                chat_history=[Message(role=Role.SYSTEM, content="respond like an echo")],
            ),
            EpisodeSpec(
                episode_id="ep-2",
                task_tag="echo",
                chat_history=[Message(role=Role.SYSTEM, content="respond like an echo")],
            ),
        ]
        if limit is not None:
            return turns[:limit]
        return turns


class _LengthMetric(BaseMetric):
    info = MetricInfo(
        name=METRIC_NAME,
        supported_tasks={"echo"},
        category="debug",
    )

    def evaluate(self, episode: EpisodeArtifact) -> MetricValue:  # type: ignore[override]
        content = episode.turns[-1].content if episode.turns else ""
        metric_value = MetricValue(metric_name=self.info.name, values=[float(len(content))])
        episode.metric_values[self.info.name] = metric_value
        return metric_value


def _register_test_components() -> None:
    try:
        registry.get(REGISTRY_GROUP_USER_PROXIES, ADAPTER_NAME)
    except RegistryError:
        register_user_proxy(name=ADAPTER_NAME, metadata=_EchoAdapter.info)(_EchoAdapter)

    try:
        registry.get(REGISTRY_GROUP_DATASETS, DATASET_NAME)
    except RegistryError:
        register_dataset(name=DATASET_NAME, metadata=_StaticDataset.info)(_StaticDataset)

    try:
        registry.get(REGISTRY_GROUP_METRICS, METRIC_NAME)
    except RegistryError:
        register_metric(name=METRIC_NAME, metadata=_LengthMetric.info)(_LengthMetric)


_register_test_components()


@pytest.fixture()
def sample_plan_manifest() -> PlanManifest:
    proxy = UserProxySpec(name="echo-proxy", adapter=ADAPTER_NAME)
    dataset = DatasetSpec(name=DATASET_NAME, split="test")
    metric = MetricSpec(name=METRIC_NAME)
    unit = EvalUnit(
        proxy_name=proxy.name,
        dataset_name=dataset.name,
        metric_name=metric.name,
        seed=0,
    )
    return PlanManifest(
        user_proxies=[proxy],
        datasets=[dataset],
        metrics=[metric],
        metric_judges={},
        config_hash="unit-test-hash",
        units=[unit],
    )


@pytest.fixture()
def temp_paths(tmp_path: Path) -> Paths:
    base = tmp_path / "mirrorbench"
    base.mkdir()
    return Paths(base)


def _unit_executor(unit: EvalUnit, controller: RunController) -> None:
    controller.on_episode_start(unit, "ep-001")
    episode = EpisodeResult(
        unit=unit,
        episode_id="ep-001",
        status=STATUS_COMPLETED,
        metric_values={
            unit.metric_name: MetricValue(
                metric_name=unit.metric_name,
                values=[0.5],
                metadata={"source": "test"},
            )
        },
        telemetry_summary={"tokens_input": 10},
        duration_s=1.2,
    )
    controller.on_episode_end(unit, episode)
    aggregate = MetricAggregate(metric_name=unit.metric_name, mean=0.5, sample_size=1)
    controller.on_metric_aggregate(unit, aggregate)
    controller.on_unit_telemetry(unit, {"tokens_input": 10, "total_response_time": 1.2})


def test_sync_backend_execution(sample_plan_manifest: PlanManifest, temp_paths: Paths) -> None:
    run_id = "run-sync"
    run_config = RunConfig()

    db_path = temp_paths.run_db_path(run_id)
    db = SQLiteRunDatabase(db_path, run_id)
    manifest_io = ManifestIO(temp_paths)

    controller = RunController(
        run_id=run_id,
        plan_manifest=sample_plan_manifest,
        run_config=run_config,
        paths=temp_paths,
        db=db,
        manifest_io=manifest_io,
        scorecards=[
            ScorecardConfig(
                name="mirror_scorecard",
                weights={sample_plan_manifest.metrics[0].name: 1.0},
            )
        ],
    )

    backend = SyncExecutionBackend(controller)
    summary = backend.run(sample_plan_manifest.units, unit_executor=_unit_executor)

    assert summary.run.run_id == run_id
    assert summary.units[0].metric_name == METRIC_NAME
    assert summary.aggregates[0].mean == pytest.approx(0.5)
    assert summary.telemetry_stats["tokens_input"] == pytest.approx(10)

    # Ensure manifests were written
    assert temp_paths.plan_manifest_path(run_id).exists()
    assert temp_paths.run_manifest_path(run_id).exists()
    assert temp_paths.summary_path(run_id).exists()


def test_sync_backend_default_executor(temp_paths: Paths) -> None:
    run_id = "run-default"
    job_run_config = RunConfig(seeds=[0])

    manifest = PlanManifest(
        user_proxies=[UserProxySpec(name="echo-proxy", adapter=ADAPTER_NAME)],
        datasets=[DatasetSpec(name=DATASET_NAME, split="test", task="echo")],
        metrics=[MetricSpec(name=METRIC_NAME)],
        metric_judges={},
        config_hash="default-executor",
        units=[
            EvalUnit(
                proxy_name="echo-proxy",
                dataset_name=DATASET_NAME,
                metric_name=METRIC_NAME,
                seed=0,
            )
        ],
    )

    db = SQLiteRunDatabase(temp_paths.run_db_path(run_id), run_id)
    manifest_io = ManifestIO(temp_paths)
    controller = RunController(
        run_id=run_id,
        plan_manifest=manifest,
        run_config=job_run_config,
        paths=temp_paths,
        db=db,
        manifest_io=manifest_io,
    )

    backend = SyncExecutionBackend(controller)
    summary = backend.run(manifest.units)

    assert summary.run.run_id == run_id
    assert summary.aggregates
    assert summary.aggregates[0].metric_name == METRIC_NAME
    assert summary.aggregates[0].sample_size >= 1
    assert summary.telemetry_stats["tokens_output"] >= 1
