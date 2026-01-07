from __future__ import annotations

from pathlib import Path

import pytest

from mirrorbench.core.config import JobConfig, RunConfig
from mirrorbench.core.constants import STATUS_CANCELLED, STATUS_COMPLETED
from mirrorbench.core.manifest import ManifestIO
from mirrorbench.core.models.plan import DatasetSpec, MetricSpec, UserProxySpec
from mirrorbench.core.models.registry import MetricInfo
from mirrorbench.core.models.run import MetricValue, RunSummary
from mirrorbench.core.registry import BaseMetric, register_metric
from mirrorbench.core.registry.components import BaseUserProxyAdapter
from mirrorbench.core.runner import Runner
from mirrorbench.io.paths import Paths
from tests.core.executor.test_sync_backend import (
    ADAPTER_NAME,
    DATASET_NAME,
    METRIC_NAME,
)

INCREMENTAL_METRIC_NAME = "metric:test/token-count"


@register_metric(
    name=INCREMENTAL_METRIC_NAME,
    metadata=MetricInfo(
        name=INCREMENTAL_METRIC_NAME,
        supported_tasks={"echo"},
        category="debug",
    ),
)
class _TokenCountMetric(BaseMetric):
    """Simple metric used to verify incremental manifests."""

    info = MetricInfo(
        name=INCREMENTAL_METRIC_NAME,
        supported_tasks={"echo"},
        category="debug",
    )

    def evaluate(self, episode):  # type: ignore[override]
        content = episode.turns[-1].content if episode.turns else ""
        value = float(len(content.split())) if content else 0.0
        metric_value = episode.metric_values.get(self.info.name)
        if metric_value is None:
            metric_value = MetricValue(metric_name=self.info.name, values=[])
            episode.metric_values[self.info.name] = metric_value
        metric_value.values.append(value)
        return metric_value


@pytest.fixture()
def temp_paths(tmp_path: Path) -> Paths:
    base = tmp_path / "mirrorbench"
    base.mkdir()
    return Paths(base)


def _make_job_config(metric_names: list[str]) -> JobConfig:
    return JobConfig(
        run=RunConfig(),
        user_proxies=[UserProxySpec(name="echo-proxy", adapter=ADAPTER_NAME)],
        datasets=[DatasetSpec(name=DATASET_NAME, split="test")],
        metrics=[MetricSpec(name=name) for name in metric_names],
    )


def test_runner_executes_from_job_config(temp_paths: Paths) -> None:
    job_cfg = _make_job_config([METRIC_NAME])
    run_id = "run-basic"

    runner = Runner.from_job_config(job_cfg, run_id=run_id, paths=temp_paths)
    assert len(runner.pending_units()) == 1
    summary = runner.run()

    assert isinstance(summary, RunSummary)
    assert summary.run.run_id == run_id
    assert len(summary.aggregates) == 1
    assert summary.aggregates[0].metric_name == METRIC_NAME

    manifest = ManifestIO(temp_paths).load_run(run_id)
    assert manifest.extras.get("revision") == 0
    executed_units = manifest.extras.get("executed_units") or []
    assert len(executed_units) == 1
    assert METRIC_NAME in executed_units[0]
    history = manifest.extras.get("history") or []
    assert len(history) == 1

    plan_revision = temp_paths.run_dir(run_id) / "plan_manifest.rev0.json"
    assert plan_revision.exists()


def test_runner_resume_skips_completed_units(temp_paths: Paths) -> None:
    job_cfg = _make_job_config([METRIC_NAME])
    run_id = "run-resume"

    first_runner = Runner.from_job_config(job_cfg, run_id=run_id, paths=temp_paths)
    first_runner.run()

    resume_runner = Runner.from_job_config(job_cfg, run_id=run_id, paths=temp_paths, resume=True)
    assert not resume_runner.pending_units()
    summary = resume_runner.run()

    manifest = ManifestIO(temp_paths).load_run(run_id)
    assert manifest.extras.get("revision") == 1
    assert manifest.extras.get("executed_units") == []
    history = manifest.extras.get("history") or []
    assert len(history) == HISTORY_ENTRIES

    assert summary.run.extra.get("status") == STATUS_COMPLETED


def test_runner_incremental_manifest_adds_metric(temp_paths: Paths) -> None:
    base_job = _make_job_config([METRIC_NAME])
    run_id = "run-incremental"

    initial_runner = Runner.from_job_config(base_job, run_id=run_id, paths=temp_paths)
    initial_runner.run()

    incremental_job = _make_job_config([METRIC_NAME, INCREMENTAL_METRIC_NAME])
    incremental_runner = Runner.from_job_config(
        incremental_job,
        run_id=run_id,
        paths=temp_paths,
        resume=True,
    )
    pending_ids = [unit.unit_id() for unit in incremental_runner.pending_units()]
    assert any(INCREMENTAL_METRIC_NAME in uid for uid in pending_ids)
    assert all(METRIC_NAME not in uid for uid in pending_ids)

    summary = incremental_runner.run()
    assert len(summary.aggregates) >= AGGREGATE_MIN_COUNT

    manifest = ManifestIO(temp_paths).load_run(run_id)
    assert manifest.extras.get("revision") == 1
    executed = manifest.extras.get("executed_units") or []
    assert any(INCREMENTAL_METRIC_NAME in uid for uid in executed)
    history = manifest.extras.get("history") or []
    assert len(history) == HISTORY_ENTRIES


def test_runner_cancellation(temp_paths: Paths) -> None:
    job_cfg = _make_job_config([METRIC_NAME, INCREMENTAL_METRIC_NAME])
    run_id = "run-cancel"

    runner = Runner.from_job_config(job_cfg, run_id=run_id, paths=temp_paths)
    progress_units: list[str] = []

    summary = runner.run(
        progress_callback=lambda unit: progress_units.append(unit.unit_id()),
        cancel_callback=lambda: len(progress_units) >= 1,
    )

    assert summary.run.extra.get("status") == STATUS_CANCELLED
    assert len(progress_units) == 1

    manifest = ManifestIO(temp_paths).load_run(run_id)
    assert manifest.extras.get("status") == STATUS_CANCELLED
    executed = manifest.extras.get("executed_units") or []
    assert len(executed) == 1
    assert progress_units[0] in executed[0]


def test_runner_validate_runtime_invokes_adapter(
    temp_paths: Paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_cfg = _make_job_config([METRIC_NAME])
    run_id = "run-validate"
    runner = Runner.from_job_config(job_cfg, run_id=run_id, paths=temp_paths)

    calls: list[tuple[str, str]] = []

    original = BaseUserProxyAdapter.validate_runtime

    def _mock_validate_runtime(self, *, config, run_id: str) -> None:  # type: ignore[no-untyped-def]
        calls.append((config.name, run_id))
        original(self, config=config, run_id=run_id)

    monkeypatch.setattr(
        BaseUserProxyAdapter, "validate_runtime", _mock_validate_runtime, raising=False
    )

    runner.validate_runtime()
    assert calls == [("echo-proxy", run_id)]


HISTORY_ENTRIES = 2
AGGREGATE_MIN_COUNT = 2
