#!/usr/bin/env python3
"""
Aggregate results from Exp 5 (Multi-Seed Main Comparison) after runs complete.

Usage (from repo root):
    python rebuttal_experiments/exp5_multiseed_main/aggregate_results.py

Reports mean ± SD across seeds per (proxy, dataset, metric) and confirms
ranking stability — extending Figure 7 from ChatbotArena-only to all datasets.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = Path.home() / ".local/share/mirrorbench/runs"
OUTPUT_DIR = Path(__file__).parent

PROXIES = ["gpt-4o", "claude-4-sonnet", "gemini-2.5-pro"]
# Extra seed datasets (ChatbotArena already covered by existing 5-seed runs)
NEW_DATASETS = ["clariq_mirror", "oasst1_mirror", "qulac_mirror"]
# Original seed=0 runs (from paper) — for including in the full table
PAPER_DATASETS = ["chatbot_arena_mirror"] + NEW_DATASETS

EXTRA_SEEDS = [42, 123, 456]  # from Exp 5 configs
PAPER_SEED = 0

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


def find_run_db(name: str) -> Path | None:
    candidate = RUNS_DIR / name / "run.db"
    if candidate.exists():
        return candidate
    return None


def query_per_seed_metrics(db_path: Path) -> dict[int, dict[str, float]]:
    """Return {seed: {metric: mean_score}} from the DB.

    Multi-seed runs store one row per (unit_id) in metrics table.
    unit_id encodes the seed as the last '|' segment.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT unit_id, metric, mean FROM metrics"
    ).fetchall()
    conn.close()

    seed_results: dict[int, dict[str, float]] = {}
    for unit_id, metric, mean_val in rows:
        if metric not in JUDGE_METRICS:
            continue
        # unit_id format: proxy|dataset|split|metric|seed
        parts = unit_id.split("|")
        try:
            seed = int(parts[-1])
        except (IndexError, ValueError):
            seed = 0
        seed_results.setdefault(seed, {})[metric] = mean_val
    return seed_results


def main() -> None:
    # Collect all data
    # For new datasets: multi-seed run with seeds [42, 123, 456]
    # For chatbot_arena: use existing 5-seed runs (seeds [0,1,2,3,4])
    all_data: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    # structure: proxy -> dataset -> seed -> {metric: score}

    missing = []

    for proxy in PROXIES:
        all_data[proxy] = {}

        # New datasets from Exp 5 runs
        for dataset in NEW_DATASETS:
            run_name = f"exp5_multiseed_{proxy.replace('-', '_')}_{dataset}"
            db = find_run_db(run_name)
            if db is None:
                missing.append(run_name)
                all_data[proxy][dataset] = {}
            else:
                all_data[proxy][dataset] = query_per_seed_metrics(db)

        # ChatbotArena: existing 5-seed run
        cb_run = f"chatbot_arena_mirror-{proxy}-user-gpt-4o-assistant-claude-4-sonnet-judge-5seed"
        # Also check alternate naming
        cb_run_alt = f"chatbot_arena_mirror-{proxy.replace('-', '_')}-user-gpt-4o-assistant-claude-4-sonnet-judge-5seed"
        db = find_run_db(cb_run) or find_run_db(cb_run_alt)
        if db:
            all_data[proxy]["chatbot_arena_mirror"] = query_per_seed_metrics(db)
        else:
            # Fall back to single-seed paper run
            paper_run = f"chatbot_arena_mirror-{proxy}-user-gpt-4o-assistant-claude-4-sonnet-judge"
            db = find_run_db(paper_run)
            if db:
                all_data[proxy]["chatbot_arena_mirror"] = query_per_seed_metrics(db)
            else:
                missing.append(cb_run)
                all_data[proxy]["chatbot_arena_mirror"] = {}

    if missing:
        print(f"[WARN] {len(missing)} runs not found:")
        for r in missing:
            print(f"  - {r}")
        print()

    # ── Print mean ± SD across seeds ──────────────────────────────────────────
    for metric in JUDGE_METRICS:
        print(f"\n{'='*80}")
        print(f"Metric: {METRIC_LABELS[metric]} — Mean ± SD across seeds")
        print(f"{'='*80}")
        header = f"  {'Proxy':<22}" + "".join(f" {ds.replace('_mirror',''):<20}" for ds in PAPER_DATASETS)
        print(header)
        print("  " + "-" * (22 + 20 * len(PAPER_DATASETS)))

        for proxy in PROXIES:
            row = f"  {proxy:<22}"
            for dataset in PAPER_DATASETS:
                seed_data = all_data.get(proxy, {}).get(dataset, {})
                scores = [v[metric] for v in seed_data.values() if metric in v]
                if not scores:
                    row += f" {'[missing]':<20}"
                elif len(scores) == 1:
                    row += f" {scores[0]:.3f}(n=1)          "
                else:
                    mu = float(np.mean(scores))
                    sd = float(np.std(scores, ddof=1))
                    row += f" {mu:.3f}±{sd:.4f}(n={len(scores)})   "
            print(row)

    # ── Ranking stability ─────────────────────────────────────────────────────
    print("\n=== Ranking Stability: consistent proxy order across all seeds? ===")
    for metric in JUDGE_METRICS:
        print(f"\n  {METRIC_LABELS[metric]}:")
        for dataset in PAPER_DATASETS:
            seed_data = {proxy: all_data.get(proxy, {}).get(dataset, {})
                         for proxy in PROXIES}
            # collect all seeds present
            all_seeds = set()
            for proxy in PROXIES:
                all_seeds.update(seed_data[proxy].keys())

            rankings_per_seed = []
            for seed in sorted(all_seeds):
                scores = []
                for proxy in PROXIES:
                    v = seed_data[proxy].get(seed, {}).get(metric)
                    if v is not None:
                        scores.append((proxy, v))
                if len(scores) == len(PROXIES):
                    scores.sort(key=lambda x: -x[1])
                    rankings_per_seed.append([p for p, _ in scores])

            if not rankings_per_seed:
                print(f"    {dataset.replace('_mirror', '')}: [no data]")
                continue

            stable = all(r == rankings_per_seed[0] for r in rankings_per_seed)
            unique_rankings = set(tuple(r) for r in rankings_per_seed)
            print(f"    {dataset.replace('_mirror','')}: "
                  f"{'STABLE ✓' if stable else f'VARIES ({len(unique_rankings)} orders)'} "
                  f"across {len(rankings_per_seed)} seeds "
                  f"| Canonical: {' > '.join(rankings_per_seed[0] if rankings_per_seed else [])}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    records = []
    for proxy in PROXIES:
        for dataset in PAPER_DATASETS:
            seed_data = all_data.get(proxy, {}).get(dataset, {})
            for seed, metric_vals in seed_data.items():
                for metric, score in metric_vals.items():
                    records.append({
                        "proxy": proxy, "dataset": dataset,
                        "seed": seed, "metric": metric,
                        "metric_label": METRIC_LABELS.get(metric, metric),
                        "score": score,
                    })

    out_path = OUTPUT_DIR / "aggregated_results.json"
    with out_path.open("w") as f:
        json.dump(records, f, indent=2)
    print(f"\nResults saved to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
