"""Main pipeline for human correlation analysis.

This script orchestrates the complete workflow:
1. Sample episodes from a run
2. Export samples for human annotation
3. Compute correlations after annotation

Usage:
    # Step 1: Export samples for annotation
    python pipeline.py export <run_id> [--num-samples 100] [--gteval-seed 42] [--pi-seed 43]

    # Step 2: Human annotators fill in labels in the generated JSON files

    # Step 3: Compute correlations
    python pipeline.py analyze <run_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from compute_correlation import (
    compute_gteval_correlation,
    compute_pi_correlation,
    save_correlation_report,
)
from export_gteval import export_gteval_samples
from export_pi import export_pi_samples


def export_samples(
    run_id: str,
    num_samples: int = 100,
    gteval_seed: int = 42,
    pi_seed: int = 43,
) -> None:
    """Export samples for both GTEval and PI metrics.

    Args:
        run_id: Run identifier
        num_samples: Number of samples per metric
        gteval_seed: Random seed for GTEval sampling
        pi_seed: Random seed for PI sampling
    """
    base_dir = Path(__file__).parent / "runs" / run_id
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"EXPORTING SAMPLES FOR RUN: {run_id}")
    print(f"{'='*80}\n")

    # Export GTEval samples
    print("=" * 80)
    print("EXPORTING GTEVAL SAMPLES")
    print("=" * 80)
    try:
        gteval_path = export_gteval_samples(
            run_id=run_id,
            output_dir=base_dir,
            num_samples=num_samples,
            seed=gteval_seed,
        )
        print(f"✓ GTEval samples exported to: {gteval_path}")
    except Exception as e:
        print(f"✗ Failed to export GTEval samples: {e}")

    print()

    # Export PI samples
    print("=" * 80)
    print("EXPORTING PI PAIRWISE SAMPLES")
    print("=" * 80)
    try:
        pi_path = export_pi_samples(
            run_id=run_id,
            output_dir=base_dir,
            num_samples=num_samples,
            seed=pi_seed,
        )
        print(f"✓ PI samples exported to: {pi_path}")
    except Exception as e:
        print(f"✗ Failed to export PI samples: {e}")

    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(f"1. Annotate the samples in: {base_dir}")
    print("   - GTEval: Edit gteval_samples.json, fill in 'human_label' (0.0 to 1.0)")
    print("   - PI: Edit pi_pairwise_samples.json, fill in 'human_label' ('A', 'B', or 'Tie')")
    print("   - Note: Judge scores are in separate files and NOT visible to annotator")
    print()
    print(f"2. Rename the files to indicate they're labeled:")
    print(f"   - mv {base_dir}/gteval_samples.json {base_dir}/gteval_labeled.json")
    print(f"   - mv {base_dir}/pi_pairwise_samples.json {base_dir}/pi_pairwise_labeled.json")
    print(f"   - Keep judge score files unchanged (gteval_judge_scores.json, pi_pairwise_judge_scores.json)")
    print()
    print(f"3. Run correlation analysis:")
    print(f"   python pipeline.py analyze {run_id}")
    print()


def analyze_samples(run_id: str) -> None:
    """Compute correlations from labeled samples.

    Args:
        run_id: Run identifier
    """
    base_dir = Path(__file__).parent / "runs" / run_id

    if not base_dir.exists():
        print(f"Error: Run directory not found: {base_dir}")
        return

    print(f"\n{'='*80}")
    print(f"ANALYZING LABELED SAMPLES FOR RUN: {run_id}")
    print(f"{'='*80}\n")

    gteval_labeled = base_dir / "gteval_labeled.json"
    gteval_judge = base_dir / "gteval_judge_scores.json"
    pi_labeled = base_dir / "pi_pairwise_labeled.json"
    pi_judge = base_dir / "pi_pairwise_judge_scores.json"

    gteval_result = None
    pi_result = None

    # Compute GTEval correlation
    if gteval_labeled.exists() and gteval_judge.exists():
        print("=" * 80)
        print("COMPUTING GTEVAL CORRELATION")
        print("=" * 80)
        try:
            gteval_result = compute_gteval_correlation(gteval_labeled, gteval_judge)
            print(f"✓ GTEval correlation computed (n={gteval_result.num_samples})")
            print(f"  Spearman ρ = {gteval_result.spearman_rho:.3f}")
        except Exception as e:
            print(f"✗ Failed to compute GTEval correlation: {e}")
    else:
        if not gteval_labeled.exists():
            print(f"⚠ GTEval labeled file not found: {gteval_labeled}")
        if not gteval_judge.exists():
            print(f"⚠ GTEval judge scores not found: {gteval_judge}")

    print()

    # Compute PI correlation
    if pi_labeled.exists() and pi_judge.exists():
        print("=" * 80)
        print("COMPUTING PI CORRELATION")
        print("=" * 80)
        try:
            pi_result = compute_pi_correlation(pi_labeled, pi_judge)
            print(f"✓ PI correlation computed (n={pi_result.num_samples})")
            print(f"  Spearman ρ = {pi_result.spearman_rho:.3f}")
        except Exception as e:
            print(f"✗ Failed to compute PI correlation: {e}")
    else:
        if not pi_labeled.exists():
            print(f"⚠ PI labeled file not found: {pi_labeled}")
        if not pi_judge.exists():
            print(f"⚠ PI judge scores not found: {pi_judge}")

    print()

    # Save report
    if gteval_result or pi_result:
        print("=" * 80)
        print("SAVING CORRELATION REPORT")
        print("=" * 80)
        try:
            save_correlation_report(base_dir, gteval_result, pi_result)
        except Exception as e:
            print(f"✗ Failed to save correlation report: {e}")
    else:
        print("⚠ No results to save - no complete labeled + judge score pairs found!")


def main() -> None:
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Human correlation analysis pipeline for MirrorBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export samples for annotation
  python pipeline.py export chatbot_arena_mirror-gemini-2.5-pro-user-gpt-4o-assistant-claude-4-sonnet-judge

  # Export with custom settings
  python pipeline.py export my_run_id --num-samples 150 --gteval-seed 100 --pi-seed 101

  # Analyze labeled samples
  python pipeline.py analyze chatbot_arena_mirror-gemini-2.5-pro-user-gpt-4o-assistant-claude-4-sonnet-judge
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export samples for annotation")
    export_parser.add_argument("run_id", help="Run identifier")
    export_parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples per metric (default: 100)",
    )
    export_parser.add_argument(
        "--gteval-seed",
        type=int,
        default=42,
        help="Random seed for GTEval sampling (default: 42)",
    )
    export_parser.add_argument(
        "--pi-seed",
        type=int,
        default=43,
        help="Random seed for PI sampling (default: 43)",
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze labeled samples")
    analyze_parser.add_argument("run_id", help="Run identifier")

    args = parser.parse_args()

    if args.command == "export":
        export_samples(
            run_id=args.run_id,
            num_samples=args.num_samples,
            gteval_seed=args.gteval_seed,
            pi_seed=args.pi_seed,
        )
    elif args.command == "analyze":
        analyze_samples(run_id=args.run_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
