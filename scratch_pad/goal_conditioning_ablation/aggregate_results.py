#!/usr/bin/env python3
"""
Aggregate results from Exp 1 (Goal-Detail Ablation) after runs complete.

Usage (from repo root):
    python rebuttal_experiments/exp1_goal_ablation/aggregate_results.py

Reads run IDs from the run names embedded in the YAML configs, queries the
mirrorbench run DB, and produces a comparison table:
  condition × proxy × dataset → {GTEval, PI, RNR} mean ± CI
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

RUNS_DIR = Path.home() / ".local/share/mirrorbench/runs"
OUTPUT_DIR = Path(__file__).parent

CONDITIONS = ["goal_full", "goal_topic", "goal_none"]
PROXIES = ["gpt-4o", "claude-4-sonnet", "gemini-2.5-pro"]
DATASETS = ["chatbot_arena_mirror", "clariq_mirror"]
JUDGE_METRICS = [
    "metric:judge/gteval",
    "metric:judge/pi_pairwise",
    "metric:judge/rubric_and_reason",
]
METRIC_LABELS = {
    "metric:judge/gteval": "GTEval",
    "metric:judge/pi_pairwise": "PI",
    "metric:judge/rubric_and_reason": "RNR",
}


def run_name(condition: str, proxy: str, dataset: str) -> str:
    return f"exp1_{condition}_{proxy.replace('-', '_')}_{dataset}"


def find_run_db(name: str) -> Path | None:
    candidate = RUNS_DIR / name / "run.db"
    if candidate.exists():
        return candidate
    return None


def query_metrics(db_path: Path) -> dict[str, dict]:
    """Return {metric_name: {mean, ci, sample_size}} from the metrics table."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT metric, mean, confidence_interval, sample_size, extras FROM metrics"
    ).fetchall()
    conn.close()
    results = {}
    for metric, mean_val, ci, n, extras_json in rows:
        if metric in JUDGE_METRICS:
            results[metric] = {
                "mean": mean_val,
                "ci": ci,
                "n": n,
                "extras": json.loads(extras_json) if extras_json else {},
            }
    return results


def main() -> None:
    # Build result table: condition -> proxy -> dataset -> metric -> {mean, ci}
    table: dict = {}
    missing_runs: list[str] = []

    for condition in CONDITIONS:
        table[condition] = {}
        for proxy in PROXIES:
            table[condition][proxy] = {}
            for dataset in DATASETS:
                name = run_name(condition, proxy, dataset)
                db = find_run_db(name)
                if db is None:
                    missing_runs.append(name)
                    table[condition][proxy][dataset] = None
                else:
                    table[condition][proxy][dataset] = query_metrics(db)

    if missing_runs:
        print(f"[WARN] {len(missing_runs)} runs not found:")
        for r in missing_runs[:10]:
            print(f"  - {r}")
        if len(missing_runs) > 10:
            print(f"  ... and {len(missing_runs) - 10} more")
        print()

    # ── Print comparison table ─────────────────────────────────────────────────
    for dataset in DATASETS:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset}")
        print(f"{'='*80}")

        for metric in JUDGE_METRICS:
            metric_label = METRIC_LABELS[metric]
            print(f"\n  Metric: {metric_label}")
            header = f"  {'Proxy':<20}" + "".join(f" {c:<20}" for c in CONDITIONS)
            print(header)
            print("  " + "-" * (20 + 20 * len(CONDITIONS)))

            for proxy in PROXIES:
                row = f"  {proxy:<20}"
                for condition in CONDITIONS:
                    result = table[condition][proxy].get(dataset)
                    if result is None:
                        row += f" {'[missing]':<20}"
                    elif metric not in result:
                        row += f" {'[no metric]':<20}"
                    else:
                        m = result[metric]
                        val = f"{m['mean']:.3f}±{m['ci']:.3f}"
                        row += f" {val:<20}"
                print(row)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    # Flatten table to list of records for easy analysis
    records = []
    for condition in CONDITIONS:
        for proxy in PROXIES:
            for dataset in DATASETS:
                result = table[condition][proxy][dataset]
                if result is None:
                    continue
                for metric, vals in result.items():
                    records.append({
                        "condition": condition,
                        "proxy": proxy,
                        "dataset": dataset,
                        "metric": metric,
                        "metric_label": METRIC_LABELS.get(metric, metric),
                        "mean": vals["mean"],
                        "ci": vals["ci"],
                        "n": vals["n"],
                    })

    out_path = OUTPUT_DIR / "aggregated_results.json"
    with out_path.open("w") as f:
        json.dump(records, f, indent=2)
    print(f"\nResults saved to {out_path.relative_to(REPO_ROOT)}")

    # ── Ranking stability check ────────────────────────────────────────────────
    print("\n=== Proxy Ranking Stability across Goal Conditions ===")
    print("(Ranking order for each metric, pooled across datasets)\n")

    for metric in JUDGE_METRICS:
        print(f"  {METRIC_LABELS[metric]}:")
        for condition in CONDITIONS:
            scores = []
            for proxy in PROXIES:
                vals_list = []
                for dataset in DATASETS:
                    r = table[condition][proxy].get(dataset)
                    if r and metric in r:
                        vals_list.append(r[metric]["mean"])
                if vals_list:
                    scores.append((proxy, float(np.mean(vals_list))))
            scores.sort(key=lambda x: -x[1])
            ranking = " > ".join(f"{p} ({s:.3f})" for p, s in scores)
            print(f"    {condition:<20}: {ranking}")
        print()


if __name__ == "__main__":
    main()
