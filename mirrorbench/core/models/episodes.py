"""Episode-level specifications and runtime artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mirrorbench.core.models.messages import JudgeVerdict, Message, TurnTelemetry
from mirrorbench.core.models.run import MetricValue


@dataclass(slots=True)
class ReferenceStats:
    """Pre-computed statistics describing real user behaviour for an episode/dataset."""

    schema_version: str = "1.0"
    """Version string used to invalidate cached statistics when the schema changes."""

    metrics: dict[str, float] = field(default_factory=dict)
    """Scalar metrics (e.g., average turn length, distinct-n) derived from the dataset."""

    distributions: dict[str, Any] = field(default_factory=dict)
    """Distributional artefacts (histograms, embeddings) keyed by a descriptive name."""

    generated_at: datetime | None = None
    """Timestamp recording when the statistics were computed."""


@dataclass(slots=True)
class EpisodeSpec:
    """Immutable description of a single dataset episode."""

    episode_id: str
    """Unique identifier for the episode within its dataset."""

    task_tag: str
    """Identifier for the task this episode belongs to (e.g., 'human_likeness_score')."""

    chat_history: list[Message]
    """Conversational history provided in the ground truth data."""

    references: dict[str, Any] = field(default_factory=dict)
    """Additional artefacts (gold utterances, annotations) exposed to metrics/judges."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form metadata describing the episode (domain, language, etc.)."""

    reference_stats: ReferenceStats | None = None
    """Optional reference statistics scoped to this episode."""


@dataclass(slots=True)
class EpisodeArtifact:
    """Runtime artefact produced when an evaluation unit executes an episode."""

    spec: EpisodeSpec
    """The original episode specification."""

    turns: list[Message] = field(default_factory=list)
    """Full transcript including the proxy-generated user turns."""

    telemetry: list[TurnTelemetry] = field(default_factory=list)
    """Per-turn telemetry collected while executing the episode."""

    judge_verdicts: list[JudgeVerdict] = field(default_factory=list)
    """All judge evaluations triggered for this episode."""

    metric_values: dict[str, MetricValue] = field(default_factory=dict)
    """Per-metric :class:`MetricValue` objects keyed by metric name."""

    errors: list[str] = field(default_factory=list)
    """Non-fatal errors encountered while processing the episode."""


@dataclass(slots=True)
class EpisodeLog:
    """Lightweight summary exposed to reporting/CLI layers."""

    episode_id: str
    """Unique identifier for the episode within its dataset."""

    task_tag: str
    """Identifier for the task this episode belongs to (e.g., 'human_likeness_score')."""

    summary: str | None = None
    """Short human-readable summary of the episode outcome (e.g., "OK", "TIMEOUT")."""

    artifact_path: str | None = None
    """Filesystem path pointing to the serialized :class:`EpisodeArtifact`."""
