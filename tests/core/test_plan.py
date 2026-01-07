from __future__ import annotations

import mirrorbench.metrics  # noqa: F401 - ensure metrics register
from mirrorbench.core.config import JobConfig, RunConfig
from mirrorbench.core.models.errors import PlannerError
from mirrorbench.core.models.plan import DatasetSpec, MetricSpec, UserProxySpec
from mirrorbench.core.plan import Planner

ADAPTER_NAME = "adapter:generic/llm"
DATASET_NAME = "dataset:jsonl/chatbot_arena_mirror"
DATASET_SPLIT = "default"
METRIC_NAME = "metric:lexical/ttr"
CONFIG_HASH_LENGTH = 16


def test_planner_builds_manifest():
    cfg = JobConfig(
        run=RunConfig(seeds=[0, 1]),
        user_proxies=[UserProxySpec(name="llm-proxy", adapter=ADAPTER_NAME)],
        datasets=[DatasetSpec(name=DATASET_NAME, split=DATASET_SPLIT)],
        metrics=[MetricSpec(name=METRIC_NAME)],
    )

    planner = Planner.from_config(cfg)
    plan = planner.build()

    expected_unit_count = (
        len(cfg.user_proxies) * len(cfg.datasets) * len(cfg.metrics) * len(cfg.run.seeds)
    )
    assert len(plan.units) == expected_unit_count
    assert planner.manifest is not None
    manifest = planner.manifest
    assert len(manifest.config_hash) == CONFIG_HASH_LENGTH
    assert manifest.user_proxies[0].adapter == ADAPTER_NAME
    assert manifest.datasets[0].split == DATASET_SPLIT
    assert manifest.metrics[0].name == METRIC_NAME
    assert manifest.metric_judges == {}
    assert {unit.seed for unit in plan.units} == {0, 1}


def test_planner_raises_when_all_units_skipped():
    cfg = JobConfig(
        run=RunConfig(seeds=[0]),
        user_proxies=[UserProxySpec(name="llm-proxy", adapter=ADAPTER_NAME)],
        datasets=[DatasetSpec(name=DATASET_NAME, split="nonexistent")],
        metrics=[MetricSpec(name=METRIC_NAME)],
    )

    planner = Planner.from_config(cfg)

    try:
        planner.build()
    except PlannerError as exc:
        message = str(exc).lower()
        assert "split" in message or "skipped" in message
        assert planner.manifest is not None
        assert planner.manifest.skipped
    else:  # pragma: no cover - defensive
        raise AssertionError("PlannerError was not raised when all units are skipped")
