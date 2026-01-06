"""Run-level metadata and aggregated metric outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from mirrorbench.core.constants import STATUS_COMPLETED
from mirrorbench.core.models.plan import EvalUnit


class RunMetadata(BaseModel):
    """Metadata recorded for each run directory."""

    run_id: str
    """Stable unique identifier for this run."""

    planner_version: str | None = None
    """Version of the planner used to generate the plan manifest."""

    engine: str | None = None
    """Name of the execution engine used to run the evaluation."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    """UTC timestamp when the run was created."""

    extra: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class MetricValue:
    """Per-episode metric output with optional distribution stats."""

    metric_name: str
    """Name of the metric."""

    values: list[float] = field(default_factory=list)
    """List of raw metric values (one per episode)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Optional metadata about the metric computation."""


@dataclass(slots=True)
class MetricAggregate:
    """Aggregated statistics for a metric across episodes."""

    metric_name: str
    """Name of the metric."""

    mean: float
    """Mean of the metric values."""

    standard_deviation: float | None = None
    """Standard deviation of the metric values."""

    confidence_interval: float | None = None
    """95% confidence interval for the mean."""

    p_value: float | None = None
    """P-value from a statistical test (if applicable)."""

    sample_size: int = 0
    """Number of episodes included in the aggregation."""

    extras: dict[str, Any] = field(default_factory=dict)
    """Additional aggregation outputs (e.g., weighted scores, metadata)."""


@dataclass(slots=True)
class EpisodeResult:
    """Captured outputs for a single evaluation unit and episode."""

    unit: EvalUnit
    """The evaluation unit this episode corresponds to."""

    episode_id: str
    """Stable unique identifier for this episode."""

    status: str = STATUS_COMPLETED
    """Execution status for the episode (e.g., 'completed', 'failed')."""

    metric_values: dict[str, MetricValue] = field(default_factory=dict)
    """Mapping of metric names to their :class:`MetricValue` instances."""

    telemetry_summary: dict[str, Any] = field(default_factory=dict)
    """Summary of telemetry data (e.g., token counts, latency)."""

    artifact_path: str | None = None
    """Path to the stored episode artifact (if any)."""

    summary: str | None = None
    """Optional human-readable summary of the episode."""

    duration_s: float | None = None
    """Optional duration of the episode in seconds."""


@dataclass(slots=True)
class RunSummary:
    """Top-level summary emitted after a run completes."""

    run: RunMetadata
    """Metadata about the run."""

    units: list[EvalUnit] = field(default_factory=list)
    """List of evaluation units that were executed."""

    aggregates: list[MetricAggregate] = field(default_factory=list)
    """List of aggregated metric results."""

    episode_results: list[EpisodeResult] = field(default_factory=list)
    """List of all episode results."""

    telemetry_stats: dict[str, Any] = field(default_factory=dict)
    """Aggregated telemetry (costs, token counts, latency summaries, etc.)."""

    notes: list[str] = field(default_factory=list)
    """Optional notes or warnings about the run."""

    scorecards: list[ScorecardResult] = field(default_factory=list)
    """Computed scorecard results for the run."""


@dataclass(slots=True)
class ScorecardResult:
    """Composite score summarising multiple metrics."""

    name: str
    """Scorecard identifier."""

    score: float | None
    """Weighted score (``None`` when missing metrics)."""

    weights: dict[str, float] = field(default_factory=dict)
    """Weights applied to each metric in the scorecard."""

    missing_metrics: set[str] = field(default_factory=set)
    """Metrics absent from the run results."""

    extras: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (e.g., component scores)."""


def metric_value_to_dict(value: MetricValue) -> dict[str, Any]:
    """Convert a :class:`MetricValue` into a JSON-serializable dict."""

    return {
        "metric_name": value.metric_name,
        "values": list(value.values),
        "metadata": value.metadata,
    }


def metric_value_from_dict(payload: Mapping[str, Any]) -> MetricValue:
    """Rehydrate a :class:`MetricValue` from a serialized payload."""

    metric_name = str(payload.get("metric_name"))
    values = list(payload.get("values", []))
    metadata = dict(payload.get("metadata", {}))
    return MetricValue(metric_name=metric_name, values=values, metadata=metadata)


def scorecard_result_to_dict(result: ScorecardResult) -> dict[str, Any]:
    """Serialize a :class:`ScorecardResult` to a JSON-friendly mapping."""

    return {
        "name": result.name,
        "score": result.score,
        "weights": dict(result.weights),
        "missing_metrics": sorted(result.missing_metrics),
        "extras": dict(result.extras),
    }


def scorecard_result_from_dict(payload: Mapping[str, Any]) -> ScorecardResult:
    """Rehydrate a :class:`ScorecardResult` from a mapping."""

    name = str(payload.get("name"))
    score = payload.get("score")
    weights = dict(payload.get("weights", {}))
    missing = set(payload.get("missing_metrics", []))
    extras = dict(payload.get("extras", {}))
    return ScorecardResult(
        name=name,
        score=None if score is None else float(score),
        weights={k: float(v) for k, v in weights.items()},
        missing_metrics=missing,
        extras=extras,
    )


__all__ = [
    "EpisodeResult",
    "MetricAggregate",
    "MetricValue",
    "RunMetadata",
    "RunSummary",
    "ScorecardResult",
    "metric_value_from_dict",
    "metric_value_to_dict",
    "scorecard_result_from_dict",
    "scorecard_result_to_dict",
]
