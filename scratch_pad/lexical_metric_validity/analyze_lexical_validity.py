#!/usr/bin/env python3
"""
Exp 3: Token Length Statistics & MATTR Stability Analysis.

Addresses R3-W1: "MATTR and HD-D are known to work better for longer-form text;
unclear whether diversity scores meaningfully reflect human-likeness for short utterances."

Two analyses:
  1. Per-dataset token statistics (turns, tokens per episode, fraction ≥ MATTR window)
  2. MATTR score vs. total token count scatter — fit linear regression, report R²
     (low R² = MATTR is length-robust, which is its design goal)

Results are written to rebuttal_experiments/exp3_lexical_validity/

Usage (from repo root):
    python rebuttal_experiments/exp3_lexical_validity/analyze_lexical_validity.py
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import NamedTuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "rebuttal_experiments/exp3_lexical_validity"

# ── Dataset JSONL paths ────────────────────────────────────────────────────────
DATASETS = {
    "chatbot_arena": REPO_ROOT / "scratch_pad/data/chatbot_arena/chatbot_arena_mirror.jsonl",
    "clariq": REPO_ROOT / "scratch_pad/data/clariq/clariq_mirror.jsonl",
    "oasst1": REPO_ROOT / "scratch_pad/data/oasst1/oasst1_mirror.jsonl",
    "qulac": REPO_ROOT / "scratch_pad/data/qulac/qulac_mirror.jsonl",
}

# MATTR window size (must match what the paper uses)
MATTR_WINDOW = 50

# ── Locate runs with MATTR data ───────────────────────────────────────────────
RUNS_DIR = Path.home() / ".local/share/mirrorbench/runs"


class EpisodeMATTR(NamedTuple):
    dataset: str
    episode_id: str
    proxy_token_count: int
    human_token_count: int
    mattr_score: float | None  # None if run lacked MATTR


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Analysis 1: Token statistics from raw JSONL ───────────────────────────────

def count_user_tokens_tiktoken(turns: list[dict]) -> tuple[int, int, list[int]]:
    """Return (num_user_turns, total_tokens, per_turn_tokens) for user turns.

    Uses the GPT-4o / cl100k_base tokenizer (same as the benchmark).
    """
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o")
    user_turns = [t for t in turns if t["role"] == "user"]
    per_turn = [len(enc.encode(t["content"])) for t in user_turns]
    return len(user_turns), sum(per_turn), per_turn


def compute_token_stats(records: list[dict]) -> dict:
    """Return token statistics across all episodes in a dataset."""
    total_tokens_list: list[int] = []
    turns_list: list[int] = []
    per_turn_means: list[float] = []

    for rec in records:
        turns = rec.get("turns", [])
        n_user, total, per_turn = count_user_tokens_tiktoken(turns)
        total_tokens_list.append(total)
        turns_list.append(n_user)
        if per_turn:
            per_turn_means.append(float(np.mean(per_turn)))

    arr = np.array(total_tokens_list)
    return {
        "n_episodes": len(records),
        "mean_user_turns": float(np.mean(turns_list)),
        "mean_total_tokens": float(np.mean(arr)),
        "median_total_tokens": float(np.median(arr)),
        "p25_total_tokens": float(np.percentile(arr, 25)),
        "p75_total_tokens": float(np.percentile(arr, 75)),
        "mean_tokens_per_turn": float(np.mean(per_turn_means)),
        "pct_episodes_above_window": float(100 * np.mean(arr >= MATTR_WINDOW)),
        "pct_episodes_above_2x_window": float(100 * np.mean(arr >= 2 * MATTR_WINDOW)),
    }


# ── Analysis 2: MATTR vs token count from existing run DBs ───────────────────

def find_mattr_runs() -> list[Path]:
    """Find run DBs that have MATTR results for human (control) sequences."""
    dbs = []
    for run_dir in RUNS_DIR.iterdir():
        db = run_dir / "run.db"
        if db.exists():
            dbs.append(db)
    return dbs


def _mattr(tokens: list[int], window: int = MATTR_WINDOW) -> float | None:
    """Compute MATTR score from a flat list of token IDs.

    Mirrors the benchmark implementation exactly.
    Returns None if token list is empty.
    """
    n = len(tokens)
    if n == 0:
        return None
    if n <= window:
        return len(set(tokens)) / n
    type_counts: dict[int, int] = {}
    unique = 0
    for tok in tokens[:window]:
        type_counts[tok] = type_counts.get(tok, 0) + 1
        if type_counts[tok] == 1:
            unique += 1
    ttr_sum = unique / window
    for idx in range(window, n):
        out = tokens[idx - window]
        type_counts[out] -= 1
        if type_counts[out] == 0:
            unique -= 1
            del type_counts[out]
        inc = tokens[idx]
        type_counts[inc] = type_counts.get(inc, 0) + 1
        if type_counts[inc] == 1:
            unique += 1
        ttr_sum += unique / window
    return ttr_sum / (n - window + 1)


def load_mattr_episodes(db_path: Path, dataset_filter: str | None = None) -> list[dict]:
    """Load per-episode token lists from a run DB and compute MATTR scores.

    The benchmark stores proxy_tokens and human_tokens in episode metadata but
    computes scores lazily in aggregate(). We recompute them here from the stored
    token ID lists.

    Returns list of dicts with keys: episode_id, dataset, proxy_token_count,
    human_token_count, proxy_mattr, human_mattr.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT e.episode_id, u.dataset, e.metric_values
        FROM episodes e
        JOIN units u ON e.unit_id = u.unit_id AND e.run_id = u.run_id
        WHERE u.metric = 'metric:lexical/mattr'
        AND e.status = 'completed'
        """
    ).fetchall()
    conn.close()

    results = []
    for episode_id, dataset, metric_values_json in rows:
        if dataset_filter and dataset_filter not in dataset:
            continue
        if not metric_values_json:
            continue
        try:
            mv = json.loads(metric_values_json)
            mattr = mv.get("metric:lexical/mattr", {})
            metadata = mattr.get("metadata", {})
            proxy_token_ids: list[int] = metadata.get("proxy_tokens", [])
            human_token_ids: list[int] = metadata.get("human_tokens", [])

            # Recompute scores from token lists (same logic as BaseLexicalDiversityMetric.aggregate)
            proxy_mattr = _mattr(proxy_token_ids) if proxy_token_ids else None
            human_mattr = _mattr(human_token_ids) if human_token_ids else None

            results.append({
                "episode_id": episode_id,
                "dataset": dataset,
                "proxy_token_count": len(proxy_token_ids),
                "human_token_count": len(human_token_ids),
                "proxy_mattr": proxy_mattr,
                "human_mattr": human_mattr,
            })
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return results


def linear_regression_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, R²) for a simple linear regression."""
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coeffs[0]), float(coeffs[1]), float(r2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    # ── Analysis 1: Token statistics ─────────────────────────────────────────
    print("=== Analysis 1: Token Statistics (from raw JSONL — human reference turns) ===")
    token_stats: dict[str, dict] = {}
    for ds_name, jsonl_path in DATASETS.items():
        if not jsonl_path.exists():
            print(f"  [SKIP] {jsonl_path}")
            continue
        print(f"  Processing {ds_name}...")
        records = load_jsonl(jsonl_path)
        stats = compute_token_stats(records)
        token_stats[ds_name] = stats
        print(f"    mean_user_turns={stats['mean_user_turns']:.1f}  "
              f"mean_total_tokens={stats['mean_total_tokens']:.0f}  "
              f"pct≥{MATTR_WINDOW}={stats['pct_episodes_above_window']:.1f}%")

    report["token_stats"] = token_stats

    # ── Analysis 2: MATTR vs token count ─────────────────────────────────────
    print("\n=== Analysis 2: MATTR vs Token Count (R² for length-robustness) ===")
    all_episodes: list[dict] = []
    dbs = find_mattr_runs()
    print(f"  Scanning {len(dbs)} run DBs...")

    for db in dbs:
        episodes = load_mattr_episodes(db)
        all_episodes.extend(episodes)

    print(f"  Total episodes with MATTR data: {len(all_episodes)}")

    # Deduplicate by (dataset, episode_id) — keep first occurrence
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for ep in all_episodes:
        key = (ep["dataset"], ep["episode_id"])
        if key not in seen:
            seen.add(key)
            deduped.append(ep)
    print(f"  After deduplication: {len(deduped)} unique (dataset, episode) pairs")

    # Filter to episodes with valid human MATTR and sufficient tokens
    valid = [ep for ep in deduped
             if ep["human_mattr"] is not None
             and ep["human_token_count"] > 0]
    print(f"  Episodes with valid human MATTR: {len(valid)}")

    # ── Proxy vs human token count comparison (from DB) ──────────────────────
    print("\n=== Analysis 2b: Proxy vs Human Token Counts (from run DBs) ===")
    proxy_human_stats: dict[str, dict] = {}
    ds_buckets: dict[str, dict[str, list]] = {}
    for ep in deduped:
        # normalise dataset key: "dataset:jsonl/chatbot_arena_mirror" -> "chatbot_arena_mirror"
        ds_raw = ep["dataset"].replace("dataset:jsonl/", "").split("/")[-1]
        if ds_raw not in ds_buckets:
            ds_buckets[ds_raw] = {"proxy": [], "human": []}
        ds_buckets[ds_raw]["proxy"].append(ep["proxy_token_count"])
        ds_buckets[ds_raw]["human"].append(ep["human_token_count"])

    for ds, counts in sorted(ds_buckets.items()):
        p = np.array(counts["proxy"])
        h = np.array(counts["human"])
        stats = {
            "n": len(p),
            "proxy_mean": float(np.mean(p)),
            "proxy_median": float(np.median(p)),
            "proxy_pct_above_window": float(100 * np.mean(p >= MATTR_WINDOW)),
            "human_mean": float(np.mean(h)),
            "human_median": float(np.median(h)),
            "human_pct_above_window": float(100 * np.mean(h >= MATTR_WINDOW)),
        }
        proxy_human_stats[ds] = stats
        print(f"  {ds}: proxy_mean={stats['proxy_mean']:.0f} ({stats['proxy_pct_above_window']:.0f}%≥{MATTR_WINDOW})"
              f"  human_mean={stats['human_mean']:.0f} ({stats['human_pct_above_window']:.0f}%≥{MATTR_WINDOW})")

    report["proxy_vs_human_token_stats"] = proxy_human_stats

    mattr_vs_length: dict = {}
    if valid:
        # Pooled analysis
        token_counts = np.array([ep["human_token_count"] for ep in valid])
        mattr_scores = np.array([ep["human_mattr"] for ep in valid])
        slope, intercept, r2 = linear_regression_r2(token_counts, mattr_scores)
        mattr_vs_length["pooled"] = {
            "n": len(valid),
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
            "mean_token_count": float(np.mean(token_counts)),
            "mean_mattr": float(np.mean(mattr_scores)),
        }
        print(f"\n  Pooled R² = {r2:.4f} (slope={slope:.6f}, n={len(valid)})")
        print(f"  → Low R² confirms MATTR is length-robust." if r2 < 0.1 else
              f"  → R² = {r2:.3f} (moderate length dependence — see FINDINGS.md)")

        # Per-dataset breakdown
        datasets_present = sorted(set(ep["dataset"].split("/")[-1].replace("dataset:jsonl/", "")
                                      for ep in valid))
        for ds in datasets_present:
            ds_eps = [ep for ep in valid if ds in ep["dataset"]]
            if len(ds_eps) < 5:
                continue
            tc = np.array([ep["human_token_count"] for ep in ds_eps])
            ms = np.array([ep["human_mattr"] for ep in ds_eps])
            s, i, r = linear_regression_r2(tc, ms)
            mattr_vs_length[ds] = {"n": len(ds_eps), "slope": s, "intercept": i, "r2": r,
                                    "mean_token_count": float(np.mean(tc))}
            print(f"  {ds}: n={len(ds_eps)}, R²={r:.4f}, mean_tokens={np.mean(tc):.0f}")

    report["mattr_vs_length"] = mattr_vs_length

    # ── Write output ──────────────────────────────────────────────────────────
    out_json = OUTPUT_DIR / "analysis_results.json"
    with out_json.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults written to {out_json.relative_to(REPO_ROOT)}")

    # Human-readable table
    out_txt = OUTPUT_DIR / "token_stats_table.txt"
    with out_txt.open("w") as f:
        f.write("=== Token Statistics per Dataset ===\n\n")
        header = f"{'Dataset':<20} {'N':>6} {'AvgTurns':>10} {'AvgTokens':>10} "
        header += f"{'MedianTokens':>13} {'P25':>6} {'P75':>6} "
        header += f"{'≥{0}(%)':<10} {'≥{1}(%)':>10}".format(MATTR_WINDOW, 2 * MATTR_WINDOW)
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for ds, s in token_stats.items():
            row = (f"{ds:<20} {s['n_episodes']:>6} {s['mean_user_turns']:>10.1f} "
                   f"{s['mean_total_tokens']:>10.0f} {s['median_total_tokens']:>13.0f} "
                   f"{s['p25_total_tokens']:>6.0f} {s['p75_total_tokens']:>6.0f} "
                   f"{s['pct_episodes_above_window']:>10.1f}% "
                   f"{s['pct_episodes_above_2x_window']:>9.1f}%")
            f.write(row + "\n")
        f.write(f"\nNote: 'window' = MATTR window size = {MATTR_WINDOW} tokens\n")

        f.write("\n\n=== Proxy vs Human Token Counts (from existing paper runs) ===\n\n")
        f.write(f"{'Dataset':<25} {'N':>6} {'ProxyMean':>10} {'Proxy≥50':>10} {'HumanMean':>11} {'Human≥50':>10}\n")
        f.write("-" * 76 + "\n")
        for ds, s in sorted(proxy_human_stats.items()):
            f.write(f"{ds:<25} {s['n']:>6} {s['proxy_mean']:>10.0f} "
                    f"{s['proxy_pct_above_window']:>9.1f}% "
                    f"{s['human_mean']:>10.0f} "
                    f"{s['human_pct_above_window']:>9.1f}%\n")
        f.write(f"\nNote: Proxy sequences are what the MATTR metric primarily evaluates.\n"
                f"Human sequences serve as the normalization baseline.\n")

        f.write("\n\n=== MATTR Length-Robustness (R²) ===\n\n")
        if mattr_vs_length:
            f.write(f"{'Scope':<30} {'N':>6} {'R²':>8} {'Slope':>12} {'MeanTokens':>12}\n")
            f.write("-" * 72 + "\n")
            for scope, v in mattr_vs_length.items():
                f.write(f"{scope:<30} {v['n']:>6} {v['r2']:>8.4f} "
                        f"{v['slope']:>12.6f} {v['mean_token_count']:>12.0f}\n")
            f.write("\nLow R² (< 0.10) indicates MATTR is robust to sequence length.\n")

    print(f"Table written to {out_txt.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
