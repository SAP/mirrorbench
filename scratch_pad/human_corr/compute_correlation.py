"""Compute correlation between human labels and judge scores.

This module analyzes human-annotated samples and computes correlation metrics
(Spearman, Pearson, Kendall's tau) against LLM judge scores.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from scipy import stats


def bootstrap_correlation(
    x: list[float], y: list[float], n_bootstrap: int = 10000, confidence: float = 0.95
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for Spearman correlation.

    Args:
        x: First variable
        y: Second variable
        n_bootstrap: Number of bootstrap samples (default: 10000)
        confidence: Confidence level (default: 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    rng = np.random.RandomState(42)  # For reproducibility
    n = len(x)
    correlations = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n, size=n, replace=True)
        x_boot = [x[i] for i in indices]
        y_boot = [y[i] for i in indices]

        # Compute correlation
        rho, _ = stats.spearmanr(x_boot, y_boot)
        correlations.append(rho)

    # Compute percentile-based confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(correlations, lower_percentile)
    upper_bound = np.percentile(correlations, upper_percentile)

    return lower_bound, upper_bound


@dataclass
class CorrelationResult:
    """Results from correlation analysis."""

    metric_name: str
    num_samples: int
    human_mean: float
    human_std: float
    judge_mean: float
    judge_std: float
    spearman_rho: float
    spearman_pvalue: float
    spearman_ci_lower: float
    spearman_ci_upper: float
    pearson_r: float
    pearson_pvalue: float
    kendall_tau: float
    kendall_pvalue: float


def compute_gteval_correlation(labeled_file: Path, judge_scores_file: Path) -> CorrelationResult:
    """Compute correlation for GTEval metric (continuous 0-1 scale).

    Args:
        labeled_file: Path to the labeled JSON file (with human_label)
        judge_scores_file: Path to the judge scores JSON file

    Returns:
        CorrelationResult with all correlation statistics
    """
    # Load labeled samples
    with labeled_file.open("r", encoding="utf-8") as f:
        labeled_samples = json.load(f)

    # Load judge scores
    with judge_scores_file.open("r", encoding="utf-8") as f:
        judge_data = json.load(f)

    # Create lookup dictionary for judge scores
    judge_lookup = {item["sample_id"]: item["judge_score"] for item in judge_data}

    # Extract human labels and judge scores
    human_scores = []
    judge_scores = []

    for sample in labeled_samples:
        sample_id = sample.get("sample_id")
        human_label = sample.get("human_label")

        if human_label is None:
            print(f"Warning: Sample {sample_id} has no human label, skipping")
            continue

        if not isinstance(human_label, (int, float)):
            print(f"Warning: Sample {sample_id} has invalid human label '{human_label}', skipping")
            continue

        # Get judge score from lookup
        judge_score = judge_lookup.get(sample_id)
        if judge_score is None:
            print(f"Warning: Sample {sample_id} has no judge score, skipping")
            continue

        human_scores.append(float(human_label))
        judge_scores.append(float(judge_score))

    if len(human_scores) < 3:
        raise ValueError(f"Not enough valid samples for correlation (got {len(human_scores)})")

    # Compute correlations
    spearman = stats.spearmanr(human_scores, judge_scores)
    pearson = stats.pearsonr(human_scores, judge_scores)
    kendall = stats.kendalltau(human_scores, judge_scores)

    # Compute bootstrap confidence interval for Spearman correlation
    ci_lower, ci_upper = bootstrap_correlation(human_scores, judge_scores)

    return CorrelationResult(
        metric_name="metric:judge/gteval",
        num_samples=len(human_scores),
        human_mean=mean(human_scores),
        human_std=stdev(human_scores) if len(human_scores) > 1 else 0.0,
        judge_mean=mean(judge_scores),
        judge_std=stdev(judge_scores) if len(judge_scores) > 1 else 0.0,
        spearman_rho=spearman.statistic,
        spearman_pvalue=spearman.pvalue,
        spearman_ci_lower=ci_lower,
        spearman_ci_upper=ci_upper,
        pearson_r=pearson.statistic,
        pearson_pvalue=pearson.pvalue,
        kendall_tau=kendall.statistic,
        kendall_pvalue=kendall.pvalue,
    )


def compute_pi_correlation(labeled_file: Path, judge_scores_file: Path) -> CorrelationResult:
    """Compute correlation for Pairwise Indistinguishability metric.

    PI metric uses categorical labels (A, B, Tie) which are converted to win rates.
    The judge score represents the proxy win rate (0.0 = always loses, 1.0 = always wins,
    0.5 = ties or 50/50 split).

    Args:
        labeled_file: Path to the labeled JSON file (with human_label)
        judge_scores_file: Path to the judge scores JSON file (with ground truth)

    Returns:
        CorrelationResult with all correlation statistics
    """
    # Load labeled samples
    with labeled_file.open("r", encoding="utf-8") as f:
        labeled_samples = json.load(f)

    # Load judge scores and ground truth
    with judge_scores_file.open("r", encoding="utf-8") as f:
        judge_data = json.load(f)

    # Create lookup dictionary for judge scores and ground truth
    judge_lookup = {
        item["sample_id"]: {
            "judge_score": item["judge_score"],
            "ground_truth_synthetic_slot": item["ground_truth_synthetic_slot"],
        }
        for item in judge_data
    }

    # Convert categorical labels to numeric win values
    human_scores = []
    judge_scores = []

    for sample in labeled_samples:
        sample_id = sample.get("sample_id")
        human_label = sample.get("human_label")

        if human_label is None:
            print(f"Warning: Sample {sample_id} has no human label, skipping")
            continue

        # Get judge data from lookup
        judge_info = judge_lookup.get(sample_id)
        if judge_info is None:
            print(f"Warning: Sample {sample_id} has no judge data, skipping")
            continue

        judge_score = judge_info["judge_score"]
        synthetic_slot = judge_info["ground_truth_synthetic_slot"]

        # Normalize label
        label_upper = str(human_label).strip().upper()
        if label_upper not in {"A", "B", "TIE"}:
            print(f"Warning: Sample {sample_id} has invalid label '{human_label}', skipping")
            continue

        # Convert to win value (1.0 = synthetic won, 0.5 = tie, 0.0 = synthetic lost)
        if label_upper == synthetic_slot:
            win_value = 1.0  # Human picked synthetic
        elif label_upper == "TIE":
            win_value = 0.5
        else:
            win_value = 0.0  # Human picked real

        human_scores.append(win_value)
        judge_scores.append(float(judge_score))

    if len(human_scores) < 3:
        raise ValueError(f"Not enough valid samples for correlation (got {len(human_scores)})")

    # Compute correlations
    spearman = stats.spearmanr(human_scores, judge_scores)
    pearson = stats.pearsonr(human_scores, judge_scores)
    kendall = stats.kendalltau(human_scores, judge_scores)

    # Compute bootstrap confidence interval for Spearman correlation
    ci_lower, ci_upper = bootstrap_correlation(human_scores, judge_scores)

    return CorrelationResult(
        metric_name="metric:judge/pi_pairwise",
        num_samples=len(human_scores),
        human_mean=mean(human_scores),
        human_std=stdev(human_scores) if len(human_scores) > 1 else 0.0,
        judge_mean=mean(judge_scores),
        judge_std=stdev(judge_scores) if len(judge_scores) > 1 else 0.0,
        spearman_rho=spearman.statistic,
        spearman_pvalue=spearman.pvalue,
        spearman_ci_lower=ci_lower,
        spearman_ci_upper=ci_upper,
        pearson_r=pearson.statistic,
        pearson_pvalue=pearson.pvalue,
        kendall_tau=kendall.statistic,
        kendall_pvalue=kendall.pvalue,
    )


def format_correlation_report(results: list[CorrelationResult]) -> str:
    """Format correlation results as a human-readable report.

    Args:
        results: List of CorrelationResult objects

    Returns:
        Formatted report string
    """
    lines = ["=" * 80, "HUMAN-JUDGE CORRELATION REPORT", "=" * 80, ""]

    for result in results:
        lines.extend(
            [
                f"Metric: {result.metric_name}",
                f"Samples: {result.num_samples}",
                "",
                f"Human Scores:  Mean = {result.human_mean:.3f}, Std = {result.human_std:.3f}",
                f"Judge Scores:  Mean = {result.judge_mean:.3f}, Std = {result.judge_std:.3f}",
                "",
                "Correlation Metrics:",
                f"  Spearman's ρ:   {result.spearman_rho:.3f} (p={result.spearman_pvalue:.4f})",
                f"    95% CI:       [{result.spearman_ci_lower:.3f}, {result.spearman_ci_upper:.3f}]",
                f"  Pearson's r:    {result.pearson_r:.3f} (p={result.pearson_pvalue:.4f})",
                f"  Kendall's τ:    {result.kendall_tau:.3f} (p={result.kendall_pvalue:.4f})",
                "",
                "Interpretation:",
            ]
        )

        # Interpretation based on Spearman (robust to outliers)
        rho = abs(result.spearman_rho)
        if rho >= 0.7:
            strength = "Strong"
        elif rho >= 0.5:
            strength = "Moderate"
        elif rho >= 0.3:
            strength = "Weak"
        else:
            strength = "Very weak"

        significance = "significant" if result.spearman_pvalue < 0.05 else "not significant"

        lines.extend([f"  {strength} correlation ({significance} at α=0.05)", "", "-" * 80, ""])

    return "\n".join(lines)


def save_correlation_report(
    output_dir: Path,
    gteval_result: CorrelationResult | None = None,
    pi_result: CorrelationResult | None = None,
) -> Path:
    """Save correlation results to JSON and text report.

    Args:
        output_dir: Directory to save report files
        gteval_result: GTEval correlation result (optional)
        pi_result: PI correlation result (optional)

    Returns:
        Path to the JSON report file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    if gteval_result:
        results.append(gteval_result)
    if pi_result:
        results.append(pi_result)

    # Save JSON
    json_data = {
        "results": [
            {
                "metric_name": r.metric_name,
                "num_samples": r.num_samples,
                "human_mean": r.human_mean,
                "human_std": r.human_std,
                "judge_mean": r.judge_mean,
                "judge_std": r.judge_std,
                "spearman_rho": r.spearman_rho,
                "spearman_pvalue": r.spearman_pvalue,
                "spearman_ci_lower": r.spearman_ci_lower,
                "spearman_ci_upper": r.spearman_ci_upper,
                "pearson_r": r.pearson_r,
                "pearson_pvalue": r.pearson_pvalue,
                "kendall_tau": r.kendall_tau,
                "kendall_pvalue": r.kendall_pvalue,
            }
            for r in results
        ]
    }

    json_path = output_dir / "correlation_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    # Save text report
    txt_path = output_dir / "correlation_report.txt"
    report_text = format_correlation_report(results)
    txt_path.write_text(report_text, encoding="utf-8")

    print(f"Saved correlation report to {output_dir}")
    print(report_text)

    return json_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python compute_correlation.py <run_dir>")
        print("  run_dir should contain:")
        print("    - gteval_labeled.json + gteval_judge_scores.json")
        print("    - pi_pairwise_labeled.json + pi_pairwise_judge_scores.json")
        sys.exit(1)

    run_dir = Path(sys.argv[1])

    gteval_labeled = run_dir / "gteval_labeled.json"
    gteval_judge = run_dir / "gteval_judge_scores.json"
    pi_labeled = run_dir / "pi_pairwise_labeled.json"
    pi_judge = run_dir / "pi_pairwise_judge_scores.json"

    gteval_result = None
    pi_result = None

    if gteval_labeled.exists() and gteval_judge.exists():
        print(f"Computing GTEval correlation...")
        gteval_result = compute_gteval_correlation(gteval_labeled, gteval_judge)
    else:
        if not gteval_labeled.exists():
            print(f"GTEval labeled file not found: {gteval_labeled}")
        if not gteval_judge.exists():
            print(f"GTEval judge scores not found: {gteval_judge}")

    if pi_labeled.exists() and pi_judge.exists():
        print(f"Computing PI correlation...")
        pi_result = compute_pi_correlation(pi_labeled, pi_judge)
    else:
        if not pi_labeled.exists():
            print(f"PI labeled file not found: {pi_labeled}")
        if not pi_judge.exists():
            print(f"PI judge scores not found: {pi_judge}")

    if gteval_result or pi_result:
        save_correlation_report(run_dir, gteval_result, pi_result)
    else:
        print("No complete labeled + judge score pairs found!")
        sys.exit(1)
