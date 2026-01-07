"""Abstract interfaces for run databases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from mirrorbench.core.models.plan import EvalUnit
from mirrorbench.core.models.run import (
    EpisodeResult,
    MetricAggregate,
    RunMetadata,
    RunSummary,
    ScorecardResult,
)


class RunDatabase(ABC):
    """Abstract base class for run database implementations."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id

    @abstractmethod
    def initialize(self) -> None:
        """Create database schema and enable pragmas if necessary."""

    @abstractmethod
    def close(self) -> None:
        """Close any open resources."""

    @abstractmethod
    def record_run(self, metadata: RunMetadata) -> None:
        """Persist run-level metadata."""

    @abstractmethod
    def record_unit(self, eval_unit: EvalUnit, status: str = "pending") -> str:
        """Persist a unit entry and return its identifier."""

    @abstractmethod
    def record_episode_result(
        self,
        unit_id: str,
        episode_result: EpisodeResult,
        status: str,
        duration_s: float | None = None,
        summary: str | None = None,
    ) -> None:
        """Persist an episode result."""

    @abstractmethod
    def record_metric_aggregate(self, unit_id: str, aggregate: MetricAggregate) -> None:
        """Persist metric aggregates for a unit."""

    @abstractmethod
    def record_unit_telemetry(self, unit_id: str, telemetry: dict[str, float]) -> None:
        """Persist aggregated telemetry stats for a unit."""

    @abstractmethod
    def load_run_summary(self) -> RunSummary:
        """Load the run summary composed of stored artifacts."""

    @abstractmethod
    def iter_units(self) -> Iterable[str]:
        """Iterate over recorded unit identifiers."""

    @abstractmethod
    def get_unit_statuses(self) -> dict[str, str]:
        """Return a mapping of unit ids to their persisted execution status."""

    @abstractmethod
    def load_run_metadata(self) -> RunMetadata | None:
        """Return persisted run metadata if available."""

    @abstractmethod
    def record_scorecard(self, scorecard: ScorecardResult) -> None:
        """Persist a computed scorecard result."""

    @abstractmethod
    def load_scorecards(self) -> list[ScorecardResult]:
        """Return all persisted scorecard results for the run."""
