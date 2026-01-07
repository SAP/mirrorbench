"""Create publication-ready correlation figure for judge validation.

This script creates a compact figure suitable for single-column paper layout,
showing scatter plots with regression lines for both GTEval and PI metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde


def load_correlation_data(run_dir: Path, metric_type: str) -> tuple[list[float], list[float]]:
    """Load human labels and judge scores for a metric.

    Args:
        run_dir: Directory containing the run data
        metric_type: Either "gteval" or "pi_pairwise"

    Returns:
        Tuple of (human_scores, judge_scores)
    """
    if metric_type == "gteval":
        labeled_file = run_dir / "gteval_labeled.json"
        judge_file = run_dir / "gteval_judge_scores.json"
    else:  # pi_pairwise
        labeled_file = run_dir / "pi_pairwise_labeled.json"
        judge_file = run_dir / "pi_pairwise_judge_scores.json"

    # Load labeled samples
    with labeled_file.open("r") as f:
        labeled_data = json.load(f)

    # Load judge scores
    with judge_file.open("r") as f:
        judge_data = json.load(f)

    # Create lookup
    if metric_type == "gteval":
        judge_lookup = {item["sample_id"]: item["judge_score"] for item in judge_data}
    else:  # pi_pairwise
        judge_lookup = {
            item["sample_id"]: {
                "judge_score": item["judge_score"],
                "ground_truth": item["ground_truth_synthetic_slot"],
            }
            for item in judge_data
        }

    human_scores = []
    judge_scores = []

    for sample in labeled_data:
        sample_id = sample["sample_id"]
        human_label = sample.get("human_label")

        if human_label is None:
            continue

        if metric_type == "gteval":
            if not isinstance(human_label, (int, float)):
                continue
            judge_score = judge_lookup.get(sample_id)
            if judge_score is None:
                continue
            human_scores.append(float(human_label))
            judge_scores.append(float(judge_score))
        else:  # pi_pairwise
            judge_info = judge_lookup.get(sample_id)
            if judge_info is None:
                continue

            # Convert categorical to win rate
            label_upper = str(human_label).strip().upper()
            if label_upper not in {"A", "B", "TIE"}:
                continue

            synthetic_slot = judge_info["ground_truth"]
            if label_upper == synthetic_slot:
                win_value = 1.0
            elif label_upper == "TIE":
                win_value = 0.5
            else:
                win_value = 0.0

            human_scores.append(win_value)
            judge_scores.append(float(judge_info["judge_score"]))

    return human_scores, judge_scores


def calculate_density_sizes(
    x: list[float], y: list[float], min_size: float = 20, max_size: float = 200
) -> np.ndarray:
    """Calculate point sizes based on local density.

    Args:
        x: X coordinates
        y: Y coordinates
        min_size: Minimum point size
        max_size: Maximum point size

    Returns:
        Array of point sizes based on density
    """
    # Stack coordinates for KDE
    xy = np.vstack([x, y])

    # Calculate density at each point using Gaussian KDE
    kde = gaussian_kde(xy)
    density = kde(xy)

    # Normalize density to size range
    density_min = density.min()
    density_max = density.max()

    if density_max > density_min:
        # Map density to size range (higher density = larger points)
        normalized_density = (density - density_min) / (density_max - density_min)
        sizes = min_size + normalized_density * (max_size - min_size)
    else:
        # All points have same density, use average size
        sizes = np.full_like(density, (min_size + max_size) / 2)

    return sizes


def create_correlation_figure(run_dir: Path, output_path: Path) -> None:
    """Create a compact correlation figure for both metrics.

    Args:
        run_dir: Directory containing the labeled data
        output_path: Path to save the figure
    """
    # Load data for both metrics
    gteval_human, gteval_judge = load_correlation_data(run_dir, "gteval")
    pi_human, pi_judge = load_correlation_data(run_dir, "pi_pairwise")

    # Compute all correlations
    gteval_spearman = stats.spearmanr(gteval_human, gteval_judge)
    gteval_pearson = stats.pearsonr(gteval_human, gteval_judge)
    gteval_kendall = stats.kendalltau(gteval_human, gteval_judge)

    pi_spearman = stats.spearmanr(pi_human, pi_judge)
    pi_pearson = stats.pearsonr(pi_human, pi_judge)
    pi_kendall = stats.kendalltau(pi_human, pi_judge)

    # Create figure with 2 subplots side-by-side
    # Increased height from 2.5 to 3.0, removed suptitle for more space
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0))

    # Set style with professional color palette
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "scatter": "#2E86AB",      # Professional blue (slightly deeper than default)
        "line": "#E63946",         # Vibrant coral red (stands out but not harsh)
        "perfect": "#989EA3",      # Muted gray for reference line
        "text_bg": "#F8F9FA",      # Very light gray background
        "text_border": "#ADB5BD",  # Medium gray border
    }

    # --- GTEval Plot ---
    # Calculate density-based sizes
    gteval_sizes = calculate_density_sizes(gteval_judge, gteval_human, min_size=5, max_size=120)

    ax1.scatter(
        gteval_judge,
        gteval_human,
        alpha=0.65,
        s=gteval_sizes,
        color=colors["scatter"],
        edgecolors="white",
        linewidth=0.8,
    )

    # Add regression line
    z = np.polyfit(gteval_judge, gteval_human, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(gteval_judge), max(gteval_judge), 100)
    ax1.plot(x_line, p(x_line), "-", color=colors["line"], linewidth=2.5, alpha=0.85, label="Fit")

    # Add diagonal reference line (perfect correlation)
    ax1.plot([0, 1], [0, 1], "--", color=colors["perfect"], linewidth=1.5, alpha=0.6, label="y=x")

    # Labels and title
    ax1.set_xlabel("Judge Score", fontsize=10)
    ax1.set_ylabel("Human Score", fontsize=10)
    ax1.set_title("GTEval", fontsize=12, pad=10)

    # Add correlation text with all three metrics (no asterisks)
    corr_text = (
        f"ρ = {gteval_spearman.statistic:.3f}\n"
        f"r = {gteval_pearson.statistic:.3f}\n"
        f"τ = {gteval_kendall.statistic:.3f}"
    )
    ax1.text(
        0.05,
        0.95,
        corr_text,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=colors["text_bg"],
            alpha=0.95,
            edgecolor=colors["text_border"],
            linewidth=1.2,
        ),
        family="monospace",
    )

    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # --- PI Pairwise Plot ---
    # Calculate density-based sizes
    pi_sizes = calculate_density_sizes(pi_judge, pi_human, min_size=5, max_size=120)

    ax2.scatter(
        pi_judge, pi_human, alpha=0.65, s=pi_sizes, color=colors["scatter"], edgecolors="white", linewidth=0.8
    )

    # Add regression line
    z = np.polyfit(pi_judge, pi_human, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(pi_judge), max(pi_judge), 100)
    ax2.plot(x_line, p(x_line), "-", color=colors["line"], linewidth=2.5, alpha=0.85, label="Fit")

    # Add diagonal reference line
    ax2.plot([0, 1], [0, 1], "--", color=colors["perfect"], linewidth=1.5, alpha=0.6, label="y=x")

    # Labels and title
    ax2.set_xlabel("Judge Score", fontsize=10)
    ax2.set_ylabel("Human Score", fontsize=10)
    ax2.set_title("PI", fontsize=12, pad=10)

    # Add correlation text with all three metrics (no asterisks)
    corr_text = (
        f"ρ = {pi_spearman.statistic:.3f}\n"
        f"r = {pi_pearson.statistic:.3f}\n"
        f"τ = {pi_kendall.statistic:.3f}"
    )
    ax2.text(
        0.05,
        0.95,
        corr_text,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=colors["text_bg"],
            alpha=0.95,
            edgecolor=colors["text_border"],
            linewidth=1.2,
        ),
        family="monospace",
    )

    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    # Removed overall title for more plot space
    # Adjust layout - no rect needed since we removed suptitle
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved correlation figure to: {output_path}")

    # Also save as PDF for LaTeX papers
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved PDF version to: {pdf_path}")

    plt.close()

    # Print statistics summary
    print("\n" + "=" * 60)
    print("CORRELATION STATISTICS")
    print("=" * 60)
    print(f"GTEval (n={len(gteval_human)}):")
    print(f"  Spearman's ρ = {gteval_spearman.statistic:.2f} (p={gteval_spearman.pvalue:.4f})")
    print(f"  Pearson's  r = {gteval_pearson.statistic:.2f} (p={gteval_pearson.pvalue:.4f})")
    print(f"  Kendall's  τ = {gteval_kendall.statistic:.2f} (p={gteval_kendall.pvalue:.4f})")
    print(f"\nPairwise Indistinguishability (n={len(pi_human)}):")
    print(f"  Spearman's ρ = {pi_spearman.statistic:.2f} (p={pi_spearman.pvalue:.4f})")
    print(f"  Pearson's  r = {pi_pearson.statistic:.2f} (p={pi_pearson.pvalue:.4f})")
    print(f"  Kendall's  τ = {pi_kendall.statistic:.2f} (p={pi_kendall.pvalue:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python create_correlation_figure.py <run_id>")
        print(
            "\nExample: python create_correlation_figure.py "
            "chatbot_arena_mirror-gemini-2.5-pro-user-gpt-4o-assistant-claude-4-sonnet-judge"
        )
        sys.exit(1)

    run_id = sys.argv[1]
    run_dir = Path(__file__).parent / "runs" / run_id

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    # Check required files
    required_files = [
        "gteval_labeled.json",
        "gteval_judge_scores.json",
        "pi_pairwise_labeled.json",
        "pi_pairwise_judge_scores.json",
    ]

    missing = [f for f in required_files if not (run_dir / f).exists()]
    if missing:
        print(f"Error: Missing required files: {', '.join(missing)}")
        sys.exit(1)

    output_path = run_dir / "judge_human_correlation_figure.png"
    create_correlation_figure(run_dir, output_path)
