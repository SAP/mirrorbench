"""Task driver that mirrors the legacy single-turn execution behaviour."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, ClassVar

from mirrorbench.core.constants import DEFAULT_TASK_DRIVER_NAME
from mirrorbench.core.models.episodes import EpisodeArtifact, EpisodeSpec
from mirrorbench.core.models.registry import TaskDriverInfo
from mirrorbench.core.registry.decorators import register_task_driver
from mirrorbench.tasks.base import EpisodeExecutionResult, TaskDriver

DEFAULT_SINGLE_TURN_DRIVER_NAME = DEFAULT_TASK_DRIVER_NAME


@register_task_driver(name=DEFAULT_SINGLE_TURN_DRIVER_NAME)
class SingleTurnTaskDriver(TaskDriver):
    """Driver that executes datasets expecting a single proxy generation per episode."""

    info: ClassVar[TaskDriverInfo] = TaskDriverInfo(
        name=DEFAULT_SINGLE_TURN_DRIVER_NAME,
        description="Single-turn driver that preserves legacy execution behaviour.",
    )

    def run_episode(
        self,
        *,
        episode: EpisodeSpec,
        proxy_session: Any,
        run_id: str,
    ) -> EpisodeExecutionResult:
        self._ensure_initialised()
        result = proxy_session.generate(turn=episode)
        turns = list(episode.chat_history)
        message = getattr(result, "message", None)
        if message is not None and message not in turns:
            turns.append(message)

        artifact = EpisodeArtifact(spec=episode, turns=turns)
        telemetry = getattr(result, "telemetry", None)
        turn_telemetries = []
        if telemetry is not None:
            artifact.telemetry.append(telemetry)
            turn_telemetries.append(telemetry)
        return EpisodeExecutionResult(artifact=artifact, turn_telemetries=turn_telemetries)

    async def run_episode_async(
        self,
        *,
        episode: EpisodeSpec,
        proxy_session: Any,
        run_id: str,
    ) -> EpisodeExecutionResult:
        self._ensure_initialised()
        async_generate = getattr(proxy_session, "generate_async", None)
        if callable(async_generate):
            response = async_generate(turn=episode)
            if inspect.isawaitable(response):
                result = await response
            else:
                result = response
        else:
            result = await asyncio.to_thread(proxy_session.generate, turn=episode)

        turns = list(episode.chat_history)
        message = getattr(result, "message", None)
        if message is not None and message not in turns:
            turns.append(message)

        artifact = EpisodeArtifact(spec=episode, turns=turns)
        telemetry = getattr(result, "telemetry", None)
        turn_telemetries = []
        if telemetry is not None:
            artifact.telemetry.append(telemetry)
            turn_telemetries.append(telemetry)
        return EpisodeExecutionResult(artifact=artifact, turn_telemetries=turn_telemetries)
