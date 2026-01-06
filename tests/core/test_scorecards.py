from __future__ import annotations

import pytest

from mirrorbench.core.config import RunConfig, ScorecardConfig
from mirrorbench.core.executor.controller import RunController
from mirrorbench.core.models.plan import PlanManifest
from mirrorbench.core.models.run import MetricAggregate
from mirrorbench.io.paths import Paths


def test_run_controller_computes_scorecards(tmp_path):
    paths = Paths(tmp_path / "mirrorbench")
    run_config = RunConfig.model_validate({})
    scorecard = ScorecardConfig(name="mirror", weights={"metric:a": 0.7, "metric:b": 0.3})
    controller = RunController(
        run_id="run-scorecard",
        plan_manifest=PlanManifest(),
        run_config=run_config,
        paths=paths,
        scorecards=[scorecard],
    )

    controller._aggregates = [
        MetricAggregate(metric_name="metric:a", mean=0.2, sample_size=3),
        MetricAggregate(metric_name="metric:b", mean=0.8, sample_size=3),
    ]

    controller.on_run_start()
    summary = controller.on_run_end()

    assert summary.scorecards
    result = summary.scorecards[0]
    assert result.name == "mirror"
    assert pytest.approx(result.score, rel=1e-6) == 0.2 * 0.7 + 0.8 * 0.3
    assert controller.db.load_scorecards()

    controller.close()
