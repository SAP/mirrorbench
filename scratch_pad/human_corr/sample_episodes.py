"""Sample episodes for human annotation from evaluation runs.

This module provides stratified sampling based on metric scores and conversation
lengths to ensure diverse coverage for correlation studies.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mirrorbench.core.models.episodes import EpisodeArtifact, EpisodeSpec, ReferenceStats
from mirrorbench.core.models.messages import JudgeVerdict, Message, Role, TurnTelemetry
from mirrorbench.core.models.run import MetricValue, metric_value_from_dict
from mirrorbench.core.run_db.sqlite import SQLiteRunDatabase
from mirrorbench.io.paths import Paths


@dataclass
class StratifiedBin:
    """Represents a single stratification bin."""

    score_range: tuple[float, float]
    length_range: tuple[int, int]
    episodes: list[SampledEpisode]


@dataclass
class SampledEpisode:
    """Container for episode data with metric scores."""

    unit_id: str
    episode_id: str
    dataset_name: str
    metric_score: float
    conversation_length: int
    artifact: EpisodeArtifact


def _deserialize_message(data: dict[str, Any]) -> Message:
    """Deserialize a Message from JSON."""
    role_str = data.get("role", "user")
    role = Role(role_str) if role_str in {"user", "assistant", "system"} else Role.USER
    return Message(
        role=role,
        content=data.get("content", ""),
        message_id=data.get("message_id"),
        name=data.get("name"),
        metadata=data.get("metadata", {}),
    )


def _deserialize_turn_telemetry(data: dict[str, Any] | None) -> TurnTelemetry | None:
    """Deserialize TurnTelemetry from JSON."""
    if data is None:
        return None
    return TurnTelemetry(
        time_to_first_token=data.get("time_to_first_token"),
        time_per_output_token=data.get("time_per_output_token"),
        total_response_time=data.get("total_response_time"),
        tokens_input=data.get("tokens_input"),
        tokens_output=data.get("tokens_output"),
        cost_usd=data.get("cost_usd"),
        provider=data.get("provider"),
        metadata=data.get("metadata", {}),
    )


def _deserialize_judge_verdict(data: dict[str, Any]) -> JudgeVerdict:
    """Deserialize a JudgeVerdict from JSON."""
    return JudgeVerdict(
        score=data.get("score"),
        label=data.get("label"),
        confidence=data.get("confidence"),
        reason=data.get("reason"),
        raw=data.get("raw", {}),
        telemetry=_deserialize_turn_telemetry(data.get("telemetry")),
    )


def _deserialize_reference_stats(data: dict[str, Any] | None) -> ReferenceStats | None:
    """Deserialize ReferenceStats from JSON."""
    if data is None:
        return None
    return ReferenceStats(
        schema_version=data.get("schema_version", "1.0"),
        metrics=data.get("metrics", {}),
        distributions=data.get("distributions", {}),
    )


def _deserialize_episode_spec(data: dict[str, Any]) -> EpisodeSpec:
    """Deserialize an EpisodeSpec from JSON."""
    return EpisodeSpec(
        episode_id=data["episode_id"],
        task_tag=data["task_tag"],
        chat_history=[_deserialize_message(msg) for msg in data.get("chat_history", [])],
        references=data.get("references", {}),
        metadata=data.get("metadata", {}),
        reference_stats=_deserialize_reference_stats(data.get("reference_stats")),
    )


def deserialize_episode_artifact(data: dict[str, Any]) -> EpisodeArtifact:
    """Deserialize an EpisodeArtifact from JSON."""
    spec = _deserialize_episode_spec(data["spec"])
    turns = [_deserialize_message(msg) for msg in data.get("turns", [])]
    telemetry = [_deserialize_turn_telemetry(t) for t in data.get("telemetry", [])]
    judge_verdicts = [_deserialize_judge_verdict(v) for v in data.get("judge_verdicts", [])]

    metric_values = {}
    for name, value_data in data.get("metric_values", {}).items():
        metric_values[name] = metric_value_from_dict(value_data)

    return EpisodeArtifact(
        spec=spec,
        turns=turns,
        telemetry=telemetry,
        judge_verdicts=judge_verdicts,
        metric_values=metric_values,
        errors=data.get("errors", []),
    )


def load_episode_artifact(paths: Paths, run_id: str, artifact_path: str) -> EpisodeArtifact:
    """Load an episode artifact from disk."""
    run_dir = paths.run_dir(run_id)
    full_path = run_dir / artifact_path
    data = json.loads(full_path.read_text(encoding="utf-8"))
    return deserialize_episode_artifact(data)


def load_episodes_for_metric(
    run_id: str,
    metric_name: str,
    paths: Paths | None = None,
) -> list[SampledEpisode]:
    """Load all episodes for a specific metric from a run.

    Args:
        run_id: Run identifier
        metric_name: Metric name (e.g., "metric:judge/gteval")
        paths: Paths instance (uses default if None)

    Returns:
        List of SampledEpisode instances
    """
    if paths is None:
        paths = Paths.default()

    db = SQLiteRunDatabase(paths.run_db_path(run_id), run_id)
    db.initialize()

    try:
        summary = db.load_run_summary()
    finally:
        db.close()

    episodes: list[SampledEpisode] = []
    for episode_result in summary.episode_results:
        if metric_name not in episode_result.metric_values:
            continue

        metric_value = episode_result.metric_values[metric_name]
        if not metric_value.values:
            continue

        # Load the artifact
        if not episode_result.artifact_path:
            continue

        artifact = load_episode_artifact(paths, run_id, episode_result.artifact_path)

        # Count conversation length (total turns)
        conv_length = len(artifact.turns)

        episodes.append(
            SampledEpisode(
                unit_id=episode_result.unit.unit_id(),
                episode_id=episode_result.episode_id,
                dataset_name=episode_result.unit.dataset_name,
                metric_score=metric_value.values[0],
                conversation_length=conv_length,
                artifact=artifact,
            )
        )

    return episodes


def stratified_sample(
    episodes: list[SampledEpisode],
    num_samples: int,
    num_score_bins: int = 3,
    num_length_bins: int = 3,
    seed: int = 42,
) -> list[SampledEpisode]:
    """Perform stratified sampling based on score and conversation length.

    Args:
        episodes: List of episodes to sample from
        num_samples: Target number of samples
        num_score_bins: Number of score range bins (default: 3 for low/med/high)
        num_length_bins: Number of conversation length bins (default: 3)
        seed: Random seed for reproducibility

    Returns:
        List of sampled episodes
    """
    if len(episodes) <= num_samples:
        return episodes

    rng = random.Random(seed)

    # Determine score and length ranges
    scores = [ep.metric_score for ep in episodes]
    lengths = [ep.conversation_length for ep in episodes]

    score_min, score_max = min(scores), max(scores)
    length_min, length_max = min(lengths), max(lengths)

    # Create bins
    score_bin_size = (score_max - score_min) / num_score_bins
    length_bin_size = max(1, (length_max - length_min) // num_length_bins)

    # Assign episodes to bins
    bins: dict[tuple[int, int], list[SampledEpisode]] = {}
    for episode in episodes:
        score_bin = min(
            num_score_bins - 1, int((episode.metric_score - score_min) / (score_bin_size + 1e-9))
        )
        length_bin = min(
            num_length_bins - 1, (episode.conversation_length - length_min) // max(1, length_bin_size)
        )
        key = (score_bin, length_bin)
        if key not in bins:
            bins[key] = []
        bins[key].append(episode)

    # Sample from each bin proportionally
    samples_per_bin = num_samples // len(bins)
    remainder = num_samples % len(bins)

    sampled: list[SampledEpisode] = []
    bin_keys = sorted(bins.keys())

    for i, key in enumerate(bin_keys):
        bin_episodes = bins[key]
        target_samples = samples_per_bin + (1 if i < remainder else 0)
        sample_size = min(target_samples, len(bin_episodes))
        sampled.extend(rng.sample(bin_episodes, sample_size))

    # If we still need more samples (due to small bins), sample from remaining
    if len(sampled) < num_samples:
        remaining = [ep for ep in episodes if ep not in sampled]
        additional_needed = num_samples - len(sampled)
        if remaining:
            sampled.extend(rng.sample(remaining, min(additional_needed, len(remaining))))

    return sampled


__all__ = [
    "SampledEpisode",
    "StratifiedBin",
    "deserialize_episode_artifact",
    "load_episode_artifact",
    "load_episodes_for_metric",
    "stratified_sample",
]
