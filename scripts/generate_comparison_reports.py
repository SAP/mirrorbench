#!/usr/bin/env python3
"""Generate comparison reports for MirrorBench runs, organized by dataset."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from model_configs import DATASETS

from mirrorbench.io.paths import Paths

CONFIG_ROOT = Path("configs")


def load_run_summary(run_name: str, paths: Paths) -> dict[str, Any] | None:
    """Load run summary for a given run name."""
    summary_path = paths.summary_path(run_name)
    if not summary_path.exists():
        return None

    try:
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[warning] failed to load summary for '{run_name}': {exc}", file=sys.stderr)
        return None


def extract_metrics_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    """Extract all metrics and scorecards from a run summary."""
    metrics: dict[str, float] = {}

    # Extract aggregates
    for agg in summary.get("aggregates", []):
        metric_name = agg.get("metric_name", "")
        mean = agg.get("mean")
        if mean is not None:
            # Shorten metric names for CSV columns
            short_name = (
                metric_name.replace("metric:", "")
                .replace("judge/", "j_")
                .replace("lexical/", "lex_")
            )
            metrics[short_name] = mean

    # Extract scorecards
    for scorecard in summary.get("scorecards", []):
        scorecard_name = scorecard.get("name", "")
        score = scorecard.get("score")
        if score is not None:
            metrics[f"sc_{scorecard_name}"] = score

    return metrics


def generate_dataset_report(dataset_key: str, paths: Paths) -> None:
    """Generate a comparison report for a specific dataset."""
    dataset_dir = CONFIG_ROOT / dataset_key
    if not dataset_dir.exists():
        print(f"[info] skipping dataset '{dataset_key}': no config directory found")
        return

    # Collect all run data for this dataset
    run_data: dict[str, dict[str, float]] = {}
    all_metric_names: set[str] = set()

    # Find all job.yaml files under this dataset
    for config_file in dataset_dir.glob("**/job.yaml"):
        try:
            with open(config_file, encoding="utf-8") as f:
                import yaml

                config = yaml.safe_load(f)

            run_name = config.get("run", {}).get("name")
            if not run_name:
                continue

            # Load the run summary
            summary = load_run_summary(run_name, paths)
            if not summary:
                continue

            # Extract metrics
            metrics = extract_metrics_from_summary(summary)
            if metrics:
                run_data[run_name] = metrics
                all_metric_names.update(metrics.keys())

        except Exception as exc:
            print(f"[warning] failed to process config '{config_file}': {exc}", file=sys.stderr)
            continue

    if not run_data:
        print(f"[info] no runs found for dataset '{dataset_key}'")
        return

    # Sort metric names for consistent ordering
    sorted_metrics = sorted(all_metric_names)

    # Write CSV report
    report_path = dataset_dir / "report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header
        header = ["run_name"] + sorted_metrics
        writer.writerow(header)

        # Write rows (sorted by run name)
        for run_name in sorted(run_data.keys()):
            metrics = run_data[run_name]
            row = [run_name]
            for metric in sorted_metrics:
                value = metrics.get(metric)
                row.append(value if value is not None else "")
            writer.writerow(row)

    print(f"[mirrorbench-script] generated report: {report_path}")
    print(f"  - {len(run_data)} runs")
    print(f"  - {len(sorted_metrics)} metrics")


def main() -> None:
    """Generate comparison reports for all datasets."""
    paths = Paths.default()

    # Process each dataset
    for dataset_spec in DATASETS:
        dataset_key = dataset_spec["key"]
        print(f"\n[mirrorbench-script] processing dataset: {dataset_key}")
        generate_dataset_report(dataset_key, paths)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        sys.exit(130)

# Usage:
# ------
# Run this script after executing jobs with generate_configs_and_run.py:
#
#   python scripts/generate_comparison_reports.py
#
# This will generate a report.csv file in each dataset directory under configs/
# For example:
#   configs/chatbot_arena_mirror/report.csv
#   configs/clariq_mirror/report.csv
#
# The CSV format is:
#   - First column: run_name (e.g., "chatbot_arena_mirror-gpt-4.1-user-gpt-4o-assistant-gpt-4o-judge")
#   - Other columns: metric values (both individual metrics and scorecards)
#
# This makes it easy to compare different user proxies, assistants, and judges for each dataset.
