from __future__ import annotations

from mirrorbench.core.constants import STATUS_COMPLETED, STATUS_FAILED
from mirrorbench.core.executor.controller import (
    _merge_aggregates,
    _merge_episode_results,
    _merge_summary_dicts,
    _merge_telemetry,
    _merge_units,
)

EXPECTED_UNIT_COUNT = 3
UPDATED_EPISODES_COUNT = 2
MERGED_TOKENS = 12.0
MERGED_LATENCY = 6.5
MERGED_EXTRA = 3
MERGED_UNIT_TOTAL = 2
MERGED_TOKENS_TOTAL = 15.0
MERGED_LATENCY_TOTAL = 1.5


def test_merge_units_deduplicates_and_sorts() -> None:
    previous = [
        {
            "proxy_name": "a",
            "dataset_name": "d1",
            "metric_name": "m1",
            "seed": 0,
        },
        {
            "proxy_name": "b",
            "dataset_name": "d1",
            "metric_name": "m2",
            "seed": 0,
        },
    ]
    current = [
        {
            "proxy_name": "b",
            "dataset_name": "d1",
            "metric_name": "m2",
            "seed": 0,
            "extra": True,
        },
        {
            "proxy_name": "c",
            "dataset_name": "d2",
            "metric_name": "m3",
            "seed": 1,
        },
    ]

    merged = _merge_units(previous, current)

    assert len(merged) == EXPECTED_UNIT_COUNT
    assert merged[0]["proxy_name"] == "a"
    assert merged[1]["proxy_name"] == "b"
    assert merged[1]["extra"] is True
    assert merged[2]["proxy_name"] == "c"


def test_merge_aggregates_updates_existing() -> None:
    previous = [{"metric_name": "m1", "mean": 0.1, "sample_size": 1}]
    current = [
        {"metric_name": "m1", "mean": 0.2, "sample_size": 2},
        {"metric_name": "m2", "mean": 0.5, "sample_size": 3},
    ]

    merged = _merge_aggregates(previous, current)

    assert merged == [
        {"metric_name": "m1", "mean": 0.2, "sample_size": 2},
        {"metric_name": "m2", "mean": 0.5, "sample_size": 3},
    ]


def test_merge_episode_results_deduplicates_by_unit_and_episode() -> None:
    previous = [
        {
            "unit": {
                "proxy_name": "a",
                "dataset_name": "d1",
                "metric_name": "m1",
                "seed": 0,
            },
            "episode_id": "ep-1",
            "status": STATUS_COMPLETED,
        }
    ]
    current = [
        {
            "unit": {
                "proxy_name": "a",
                "dataset_name": "d1",
                "metric_name": "m1",
                "seed": 0,
            },
            "episode_id": "ep-1",
            "status": STATUS_FAILED,
        },
        {
            "unit": {
                "proxy_name": "a",
                "dataset_name": "d1",
                "metric_name": "m1",
                "seed": 0,
            },
            "episode_id": "ep-2",
            "status": STATUS_COMPLETED,
        },
    ]

    merged = _merge_episode_results(previous, current)

    assert len(merged) == UPDATED_EPISODES_COUNT
    status_by_episode = {item["episode_id"]: item["status"] for item in merged}
    assert status_by_episode["ep-1"] == STATUS_FAILED
    assert status_by_episode["ep-2"] == STATUS_COMPLETED


def test_merge_telemetry_sums_numeric_values_and_overrides_others() -> None:
    previous = {"tokens": 10, "latency": 5.0, "notes": "old"}
    current = {"tokens": 2, "latency": 1.5, "notes": "new", "extra": MERGED_EXTRA}

    merged = _merge_telemetry(previous, current)

    assert merged["tokens"] == MERGED_TOKENS
    assert merged["latency"] == MERGED_LATENCY
    assert merged["notes"] == "new"
    assert merged["extra"] == MERGED_EXTRA


def test_merge_summary_dicts_combines_all_sections() -> None:
    previous = {
        "run": {"extra": {"status": STATUS_COMPLETED, "revision": 0}},
        "units": [
            {
                "proxy_name": "a",
                "dataset_name": "d1",
                "metric_name": "m1",
                "seed": 0,
            }
        ],
        "aggregates": [
            {
                "metric_name": "m1",
                "mean": 0.1,
                "sample_size": 1,
            }
        ],
        "episode_results": [
            {
                "unit": {
                    "proxy_name": "a",
                    "dataset_name": "d1",
                    "metric_name": "m1",
                    "seed": 0,
                },
                "episode_id": "ep-1",
                "status": STATUS_COMPLETED,
            }
        ],
        "telemetry_stats": {"tokens": 10},
        "notes": ["prev"],
    }
    current = {
        "run": {"extra": {"status": STATUS_COMPLETED, "revision": 1}},
        "units": [
            {
                "proxy_name": "a",
                "dataset_name": "d1",
                "metric_name": "m2",
                "seed": 0,
            }
        ],
        "aggregates": [
            {
                "metric_name": "m2",
                "mean": 0.5,
                "sample_size": 2,
            }
        ],
        "episode_results": [
            {
                "unit": {
                    "proxy_name": "a",
                    "dataset_name": "d1",
                    "metric_name": "m2",
                    "seed": 0,
                },
                "episode_id": "ep-2",
                "status": STATUS_COMPLETED,
            }
        ],
        "telemetry_stats": {"tokens": 5, "latency": 1.5},
        "notes": ["curr"],
    }

    merged = _merge_summary_dicts(previous, current)

    assert merged["run"]["extra"]["revision"] == 1
    assert len(merged["units"]) == MERGED_UNIT_TOTAL
    assert {item["metric_name"] for item in merged["aggregates"]} == {"m1", "m2"}
    assert len(merged["episode_results"]) == MERGED_UNIT_TOTAL
    assert merged["telemetry_stats"]["tokens"] == MERGED_TOKENS_TOTAL
    assert merged["telemetry_stats"]["latency"] == MERGED_LATENCY_TOTAL
    assert merged["notes"] == ["prev", "curr"]
