from __future__ import annotations

from datetime import UTC

from mirrorbench.core.models.plan import (
    DatasetSpec,
    EvalUnit,
    JudgeSpec,
    MetricSpec,
    PlanManifest,
    UserProxySpec,
)
from mirrorbench.core.models.run import RunMetadata

ADAPTER_NAME = "adapter:test/echo"
DATASET_NAME = "dataset:jsonl/qa"
METRIC_NAME = "metric:lexical/ttr"


def test_plan_manifest_roundtrip() -> None:
    user_proxy = UserProxySpec(name="demo-proxy", adapter=ADAPTER_NAME, label="Echo")
    dataset = DatasetSpec(name=DATASET_NAME, split="test", label="Dummy")
    judge = JudgeSpec(name="judge:test", params={"prompt": "p1"})
    metric = MetricSpec(name=METRIC_NAME, judge=judge)

    manifest = PlanManifest(
        user_proxies=[user_proxy],
        datasets=[dataset],
        metrics=[metric],
        metric_judges={metric.name: judge},
        config_hash="abcdef1234567890",
        units=[
            EvalUnit(
                proxy_name=user_proxy.name,
                dataset_name=dataset.name,
                metric_name=metric.name,
                seed=0,
                judge_name=judge.name,
            )
        ],
    )

    manifest_dict = manifest.model_dump()
    restored = PlanManifest.model_validate(manifest_dict)

    assert restored.user_proxies[0].name == user_proxy.name
    assert restored.metric_judges[metric.name].params["prompt"] == "p1"
    assert restored.units[0].judge_name == judge.name


def test_run_metadata_roundtrip() -> None:
    metadata = RunMetadata(
        run_id="run-123",
        planner_version="1.0.0",
        engine="sync",
        extra={"commit": "deadbeef"},
    )

    metadata_dict = metadata.model_dump()
    restored = RunMetadata.model_validate(metadata_dict)

    assert restored.run_id == metadata.run_id
    assert restored.extra == {"commit": "deadbeef"}
    assert restored.created_at.tzinfo is UTC
