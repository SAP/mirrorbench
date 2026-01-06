"""Export Pairwise Indistinguishability samples for human annotation.

PI presents two conversations (A and B) without revealing which is real/synthetic,
and asks the labeler to choose which seems more human-like (or if they're tied).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from mirrorbench.core.models.messages import Message, Role
from mirrorbench.io.paths import Paths
from mirrorbench.metrics.util import resolve_reference_conversation

from sample_episodes import SampledEpisode, load_episodes_for_metric, stratified_sample


def format_conversation(messages: list[Message]) -> list[dict[str, str]]:
    """Format a conversation as structured JSON for human labeler."""
    formatted = []
    for msg in messages:
        role_label = msg.role.value if isinstance(msg.role, Role) else str(msg.role)
        formatted.append({"role": role_label, "content": msg.content})
    return formatted


def export_pi_samples(
    run_id: str,
    output_dir: Path,
    num_samples: int = 100,
    seed: int = 43,  # Different seed from GTEval to get different samples
    paths: Paths | None = None,
) -> tuple[Path, Path]:
    """Export Pairwise Indistinguishability samples for human annotation.

    Args:
        run_id: Run identifier
        output_dir: Directory to save the annotation file
        num_samples: Number of samples to export (default: 100)
        seed: Random seed for sampling (default: 43, different from GTEval)
        paths: Paths instance (uses default if None)

    Returns:
        Tuple of (annotation_file_path, judge_scores_file_path)
    """
    if paths is None:
        paths = Paths.default()

    metric_name = "metric:judge/pi_pairwise"

    # Load all episodes for this metric
    print(f"Loading episodes for {metric_name} from run {run_id}...")
    episodes = load_episodes_for_metric(run_id, metric_name, paths)
    print(f"Found {len(episodes)} episodes")

    # Stratified sampling
    print(f"Sampling {num_samples} episodes using stratified sampling...")
    sampled = stratified_sample(episodes, num_samples, seed=seed)
    print(f"Sampled {len(sampled)} episodes")

    # Export to JSON with randomized order - separate files for annotation and judge scores
    rng = random.Random(seed)
    annotation_samples: list[dict[str, Any]] = []
    judge_scores: list[dict[str, Any]] = []

    for episode in sampled:
        artifact = episode.artifact

        # Get real conversation from references
        real_conversation = resolve_reference_conversation(artifact)
        if not real_conversation:
            print(f"Warning: No reference conversation for episode {episode.episode_id}, skipping")
            continue

        # Get synthetic conversation
        synthetic_conversation = artifact.turns

        # Randomize order (50/50 chance)
        synthetic_is_a = rng.random() < 0.5

        if synthetic_is_a:
            conversation_a = format_conversation(synthetic_conversation)
            conversation_b = format_conversation(real_conversation)
            ground_truth = "A"  # Synthetic is A
        else:
            conversation_a = format_conversation(real_conversation)
            conversation_b = format_conversation(synthetic_conversation)
            ground_truth = "B"  # Synthetic is B

        # Sample for annotation (NO judge score visible)
        annotation_samples.append(
            {
                "sample_id": f"{episode.unit_id}::{episode.episode_id}",
                "dataset_name": episode.dataset_name,
                "conversation_a": conversation_a,
                "conversation_b": conversation_b,
                "instructions": (
                    "Compare conversations A and B. Both are complete conversations between a user "
                    "and an assistant. Which conversation seems more natural and human-like in terms "
                    "of the USER's behavior? Choose 'A', 'B', or 'Tie' if they seem equally human-like."
                ),
                "human_label": None,  # To be filled by annotator: "A", "B", or "Tie"
            }
        )

        # Judge scores and ground truth (kept separate)
        judge_scores.append(
            {
                "sample_id": f"{episode.unit_id}::{episode.episode_id}",
                "judge_score": episode.metric_score,
                "ground_truth_synthetic_slot": ground_truth,
            }
        )

    # Save annotation file (for human labeling)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = output_dir / "pi_pairwise_samples.json"
    with annotation_path.open("w", encoding="utf-8") as f:
        json.dump(annotation_samples, f, indent=2, ensure_ascii=False)

    # Save judge scores file (hidden from annotator)
    judge_path = output_dir / "pi_pairwise_judge_scores.json"
    with judge_path.open("w", encoding="utf-8") as f:
        json.dump(judge_scores, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(annotation_samples)} samples to {annotation_path}")
    print(f"Saved judge scores to {judge_path}")
    return annotation_path, judge_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python export_pi.py <run_id> [num_samples] [seed]")
        sys.exit(1)

    run_id = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 43

    output_dir = Path(__file__).parent / "runs" / run_id
    export_pi_samples(run_id, output_dir, num_samples, seed)
