from __future__ import annotations

from mirrorbench.report import generate_json_report


def test_generate_json_report():
    summary = {
        "run": {"run_id": "run-123", "extra": {"status": "completed"}},
        "aggregates": [{"metric_name": "metric", "mean": 1.0}],
        "units": [
            {"proxy_name": "proxy", "dataset_name": "dataset", "metric_name": "metric", "seed": 0}
        ],
        "telemetry_stats": {"tokens": 100},
        "notes": ["ok"],
    }
    report = generate_json_report(summary)
    assert report["run"]["run_id"] == "run-123"
    assert report["aggregates"][0]["metric_name"] == "metric"
    expected_tokens = 100
    assert report["telemetry"]["tokens"] == expected_tokens
    assert report["notes"] == ["ok"]
