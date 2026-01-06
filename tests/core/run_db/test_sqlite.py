from __future__ import annotations

from pathlib import Path

from mirrorbench.core.constants import STATUS_COMPLETED, STATUS_CREATED, STATUS_RUNNING
from mirrorbench.core.models.plan import EvalUnit
from mirrorbench.core.models.run import (
    EpisodeResult,
    MetricAggregate,
    MetricValue,
    RunMetadata,
    RunSummary,
)
from mirrorbench.core.run_db.sqlite import SQLiteRunDatabase

EXPECTED_MEAN = 0.7
EXPECTED_TOKENS = 12

ADAPTER_NAME = "adapter:test/echo"
DATASET_NAME = "dataset:test/static"
METRIC_NAME = "metric:test/distinct_n"


def test_sqlite_run_database_roundtrip(tmp_path: Path) -> None:
    run_id = "run-001"
    db_path = tmp_path / "runs" / run_id / "run.db"

    db = SQLiteRunDatabase(db_path, run_id)
    db.initialize()

    metadata = RunMetadata(
        run_id=run_id,
        planner_version="1.0.0",
        engine="sync",
        extra={"status": STATUS_CREATED, "notes": ["initial setup"]},
    )
    db.record_run(metadata)

    unit = EvalUnit(
        proxy_name=ADAPTER_NAME,
        dataset_name=DATASET_NAME,
        metric_name=METRIC_NAME,
        seed=0,
        judge_name="judge-a",
    )
    unit_id = db.record_unit(unit, status=STATUS_RUNNING)

    episode_result = EpisodeResult(
        unit=unit,
        episode_id="ep-001",
        status=STATUS_COMPLETED,
        metric_values={
            METRIC_NAME: MetricValue(
                metric_name=METRIC_NAME,
                values=[0.7],
                metadata={"detail": "test"},
            )
        },
        telemetry_summary={"tokens_input": 12},
        artifact_path="artifact.json",
        summary="summary text",
        duration_s=1.5,
    )
    db.record_episode_result(unit_id, episode_result, status=STATUS_COMPLETED)

    aggregate = MetricAggregate(metric_name=METRIC_NAME, mean=EXPECTED_MEAN, sample_size=1)
    db.record_metric_aggregate(unit_id, aggregate)

    db.record_unit_telemetry(unit_id, {"tokens_input": EXPECTED_TOKENS, "total_response_time": 1.5})

    summary = db.load_run_summary()

    assert isinstance(summary, RunSummary)
    assert summary.run.run_id == run_id
    assert summary.units[0].proxy_name == ADAPTER_NAME
    assert summary.aggregates[0].mean == EXPECTED_MEAN
    assert summary.episode_results[0].summary == "summary text"
    assert summary.episode_results[0].metric_values[METRIC_NAME].values == [0.7]
    assert summary.telemetry_stats["tokens_input"] == EXPECTED_TOKENS
    assert summary.notes == ["initial setup"]

    assert list(db.iter_units()) == [unit_id]

    db.close()


def test_record_unit_preserves_episode_rows(tmp_path: Path) -> None:
    run_id = "run-dup"
    db_path = tmp_path / "runs" / run_id / "run.db"
    db = SQLiteRunDatabase(db_path, run_id)
    db.initialize()

    metadata = RunMetadata(
        run_id=run_id,
        planner_version="1.0.0",
        engine="sync",
        extra={"status": STATUS_CREATED},
    )
    db.record_run(metadata)

    unit = EvalUnit(
        proxy_name=ADAPTER_NAME,
        dataset_name=DATASET_NAME,
        metric_name=METRIC_NAME,
        seed=0,
        judge_name=None,
    )

    unit_id = db.record_unit(unit, status=STATUS_RUNNING)
    episode_result = EpisodeResult(
        unit=unit,
        episode_id="ep-001",
        status=STATUS_COMPLETED,
        metric_values={METRIC_NAME: MetricValue(metric_name=METRIC_NAME, values=[0.5])},
    )
    db.record_episode_result(unit_id, episode_result, status=STATUS_COMPLETED)

    # Update unit status: should not delete existing episode rows
    db.record_unit(unit, status=STATUS_COMPLETED)

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    assert count == 1

    db.close()
