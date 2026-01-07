from __future__ import annotations

from pathlib import Path

import pytest

from mirrorbench.core.config import RunConfig
from mirrorbench.core.manifest import ManifestIO, RunManifest
from mirrorbench.core.models.plan import (
    DatasetSpec,
    EvalUnit,
    MetricSpec,
    PlanManifest,
    UserProxySpec,
)
from mirrorbench.io.paths import Paths

ADAPTER_NAME = "adapter:test/echo"
DATASET_NAME = "dataset:test/static"
METRIC_NAME = "metric:test/distinct_n"


@pytest.fixture()
def temp_paths(tmp_path: Path) -> Paths:
    base = tmp_path / "mirrorbench"
    base.mkdir()
    return Paths(base)


def _sample_plan_manifest() -> PlanManifest:
    proxy = UserProxySpec(name="echo-proxy", adapter=ADAPTER_NAME)
    dataset = DatasetSpec(name=DATASET_NAME, split="test")
    metric = MetricSpec(name=METRIC_NAME)
    units = [
        EvalUnit(
            proxy_name=proxy.name,
            dataset_name=dataset.name,
            metric_name=metric.name,
            seed=0,
        )
    ]
    return PlanManifest(
        user_proxies=[proxy],
        datasets=[dataset],
        metrics=[metric],
        metric_judges={},
        config_hash="manifest-test",
        units=units,
    )


def test_plan_manifest_roundtrip(temp_paths: Paths) -> None:
    io = ManifestIO(temp_paths)
    manifest = _sample_plan_manifest()

    run_id = "run-123"
    path = io.write_plan(run_id, manifest)

    assert path.exists()
    loaded = io.load_plan(run_id)
    assert loaded == manifest


def test_run_manifest_roundtrip(temp_paths: Paths) -> None:
    io = ManifestIO(temp_paths)
    plan = _sample_plan_manifest()
    run_cfg = RunConfig()
    manifest = RunManifest(plan=plan, run_config=run_cfg)

    run_id = "run-456"
    path = io.write_run(run_id, manifest)

    assert path.exists()
    loaded = io.load_run(run_id)
    assert loaded.plan.user_proxies[0].name == "echo-proxy"
    assert loaded.run_config.engine == run_cfg.engine
    assert loaded.schema_version == "1.0"
