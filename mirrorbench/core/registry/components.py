"""Base classes defining the contracts for registry components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from statistics import mean
from typing import Any, ClassVar

from mirrorbench.core.models.episodes import EpisodeArtifact, EpisodeSpec, ReferenceStats
from mirrorbench.core.models.messages import JudgeVerdict
from mirrorbench.core.models.plan import DatasetSpec, UserProxySpec
from mirrorbench.core.models.registry import (
    DatasetInfo,
    JudgeInfo,
    MetricInfo,
    ModelClientInfo,
    UserProxyAdapterInfo,
)
from mirrorbench.core.models.run import MetricAggregate, MetricValue


class BaseUserProxyAdapter(ABC):
    """Base contract for user proxy adapters registered with MirrorBench."""

    info: ClassVar[UserProxyAdapterInfo]

    @abstractmethod
    def spawn(self, *, config: UserProxySpec, run_id: str) -> Any:
        """Create a session or driver for the given proxy configuration."""

    def validate_runtime(self, *, config: UserProxySpec, run_id: str) -> None:
        """Ensure required credentials and dependencies are available before execution."""

        session = self.spawn(config=config, run_id=run_id)
        validator = getattr(session, "validate_credentials", None)
        if callable(validator):
            validator()
        shutdown = getattr(session, "shutdown", None)
        if callable(shutdown):
            shutdown()
        self.shutdown()

    def shutdown(self) -> None:
        """Optional hook to release adapter resources."""

        return None


class BaseDatasetLoader(ABC):
    """Base contract for dataset loaders."""

    info: ClassVar[DatasetInfo]

    @abstractmethod
    def episodes(
        self,
        *,
        spec: DatasetSpec,
        split: str,
        limit: int | None = None,
    ) -> Iterable[EpisodeSpec]:
        """Yield episode specifications for the requested dataset split."""

    def reference_stats(self, spec: DatasetSpec) -> ReferenceStats | None:
        """Return optional reference statistics for the dataset."""

        return None


class BaseMetric(ABC):
    """Base contract for metrics."""

    info: ClassVar[MetricInfo]

    @abstractmethod
    def evaluate(self, episode: EpisodeArtifact) -> MetricValue:
        """Compute metric values for a single episode artifact."""

    def aggregate(self, values: Sequence[MetricValue]) -> MetricAggregate:
        """Aggregate per-episode metric values using a simple mean."""

        all_values: list[float] = []
        for value in values:
            all_values.extend(value.values)
        if not all_values:
            return MetricAggregate(metric_name=self.info.name, mean=0.0, sample_size=0)
        return MetricAggregate(
            metric_name=self.info.name,
            mean=mean(all_values),
            sample_size=len(all_values),
        )


class BaseJudge(ABC):
    """Base contract for LLM judges."""

    info: ClassVar[JudgeInfo]

    @abstractmethod
    def score(self, episode: EpisodeArtifact) -> JudgeVerdict:
        """Produce a normalized verdict for the given episode artifact."""


class BaseModelClient(ABC):
    """Base contract for model clients (LLM, embeddings, tools)."""

    info: ClassVar[ModelClientInfo]

    def validate_credentials(self) -> None:
        """Optional hook to verify that required credentials are present."""

        return None

    def shutdown(self) -> None:
        """Optional hook to release underlying resources (HTTP pools, etc.)."""

        return None


__all__ = [
    "BaseDatasetLoader",
    "BaseJudge",
    "BaseMetric",
    "BaseModelClient",
    "BaseUserProxyAdapter",
]
