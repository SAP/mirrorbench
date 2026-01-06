"""Base classes and runtime results for task drivers."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from mirrorbench.core.config import RunConfig
from mirrorbench.core.models.episodes import EpisodeArtifact, EpisodeSpec
from mirrorbench.core.models.errors import TaskDriverError
from mirrorbench.core.models.messages import TurnTelemetry
from mirrorbench.core.models.plan import DatasetSpec
from mirrorbench.core.models.registry import TaskDriverInfo
from mirrorbench.io.paths import Paths


@dataclass(slots=True)
class EpisodeExecutionResult:
    """Outcome returned by task drivers after executing a dataset episode."""

    artifact: EpisodeArtifact
    turn_telemetries: list[TurnTelemetry] = field(default_factory=list)


class TaskDriver(ABC):
    """Abstract base class for dataset-specific episode orchestrators."""

    info: ClassVar[TaskDriverInfo]

    def __init__(self) -> None:
        self._initialised = False

    def setup(
        self,
        *,
        run_id: str,
        run_config: RunConfig,
        paths: Paths,
        dataset: DatasetSpec,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialise the driver before executing any episodes.

        Subclasses overriding this method must call ``super().setup`` to ensure
        the initialisation flag is set. ``params`` contains dataset- or job-level
        configuration, while ``dataset`` exposes the resolved dataset spec.
        """

        self._initialised = True

    @abstractmethod
    def run_episode(
        self,
        *,
        episode: EpisodeSpec,
        proxy_session: Any,
        run_id: str,
    ) -> EpisodeExecutionResult:
        """Execute a single episode using the provided proxy session."""

    async def run_episode_async(
        self,
        *,
        episode: EpisodeSpec,
        proxy_session: Any,
        run_id: str,
    ) -> EpisodeExecutionResult:
        """Execute a single episode asynchronously, falling back to threads."""

        self._ensure_initialised()
        return await asyncio.to_thread(
            self.run_episode,
            episode=episode,
            proxy_session=proxy_session,
            run_id=run_id,
        )

    def shutdown(self) -> None:
        """Release driver resources (optional)."""

        self._initialised = False

    def _ensure_initialised(self) -> None:
        if not self._initialised:
            raise TaskDriverError(
                f"Task driver '{self.info.name}' has not been initialised via setup()"
            )
