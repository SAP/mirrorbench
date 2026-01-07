from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import ClassVar

from mirrorbench.core.config import RunConfig
from mirrorbench.core.models.episodes import EpisodeArtifact, EpisodeSpec
from mirrorbench.core.models.messages import Message, Role
from mirrorbench.core.models.plan import DatasetSpec
from mirrorbench.core.models.registry import TaskDriverInfo
from mirrorbench.io.paths import Paths
from mirrorbench.tasks.base import EpisodeExecutionResult, TaskDriver
from mirrorbench.tasks.single_turn import SingleTurnTaskDriver


class _StubDriver(TaskDriver):
    info: ClassVar[TaskDriverInfo] = TaskDriverInfo(
        name="task:test/stub",
        description="Stub driver for async tests.",
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def run_episode(
        self,
        *,
        episode: EpisodeSpec,
        proxy_session: object,
        run_id: str,
    ) -> EpisodeExecutionResult:
        _ = proxy_session, run_id
        self.calls += 1
        artifact = EpisodeArtifact(spec=episode, turns=list(episode.chat_history))
        return EpisodeExecutionResult(artifact=artifact)


class _AsyncSession:
    def __init__(self, *, content: str = "reply") -> None:
        self.async_calls = 0
        self.sync_calls = 0
        self._content = content

    def generate(self, *, turn: EpisodeSpec, request_params=None):  # type: ignore[override]
        self.sync_calls += 1
        _ = turn, request_params
        return SimpleNamespace(
            message=Message(role=Role.ASSISTANT, content=self._content), telemetry=None
        )

    async def generate_async(self, *, turn: EpisodeSpec, request_params=None):  # type: ignore[override]
        self.async_calls += 1
        _ = turn, request_params
        return SimpleNamespace(
            message=Message(role=Role.ASSISTANT, content=self._content), telemetry=None
        )


class _SyncSession:
    def __init__(self, *, content: str = "reply") -> None:
        self.sync_calls = 0
        self._content = content

    def generate(self, *, turn: EpisodeSpec, request_params=None):  # type: ignore[override]
        self.sync_calls += 1
        _ = turn, request_params
        return SimpleNamespace(
            message=Message(role=Role.ASSISTANT, content=self._content), telemetry=None
        )


def _build_episode() -> EpisodeSpec:
    return EpisodeSpec(
        episode_id="ep-1",
        task_tag="unit",
        chat_history=[Message(role=Role.USER, content="Hello")],
    )


def test_task_driver_run_episode_async_falls_back_to_thread(tmp_path) -> None:
    driver = _StubDriver()
    driver.setup(
        run_id="run-1",
        run_config=RunConfig.model_validate({}),
        paths=Paths(tmp_path / "mirrorbench"),
        dataset=DatasetSpec(name="dataset:test"),
    )

    result = asyncio.run(
        driver.run_episode_async(
            episode=_build_episode(),
            proxy_session=object(),
            run_id="run-1",
        )
    )

    assert driver.calls == 1
    assert result.artifact.turns[0].content == "Hello"


def test_single_turn_driver_run_episode_async_prefers_generate_async(tmp_path) -> None:
    driver = SingleTurnTaskDriver()
    driver.setup(
        run_id="run-1",
        run_config=RunConfig.model_validate({}),
        paths=Paths(tmp_path / "mirrorbench"),
        dataset=DatasetSpec(name="dataset:test"),
    )

    session = _AsyncSession(content="async-reply")
    result = asyncio.run(
        driver.run_episode_async(
            episode=_build_episode(),
            proxy_session=session,
            run_id="run-1",
        )
    )

    assert session.async_calls == 1
    assert session.sync_calls == 0
    assert result.artifact.turns[-1].content == "async-reply"


def test_single_turn_driver_run_episode_async_falls_back_to_generate(tmp_path) -> None:
    driver = SingleTurnTaskDriver()
    driver.setup(
        run_id="run-1",
        run_config=RunConfig.model_validate({}),
        paths=Paths(tmp_path / "mirrorbench"),
        dataset=DatasetSpec(name="dataset:test"),
    )

    session = _SyncSession(content="sync-reply")
    result = asyncio.run(
        driver.run_episode_async(
            episode=_build_episode(),
            proxy_session=session,
            run_id="run-1",
        )
    )

    assert session.sync_calls == 1
    assert result.artifact.turns[-1].content == "sync-reply"
