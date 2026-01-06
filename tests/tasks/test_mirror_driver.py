from __future__ import annotations

import asyncio
from typing import ClassVar

import pytest

import mirrorbench.datasets  # - ensure dataset registry entries
import mirrorbench.metrics  # noqa: F401 - ensure metric registry entries
from mirrorbench.adapters.llm import LLMAdapter
from mirrorbench.core.config import CacheConfig, JobConfig, RunConfig
from mirrorbench.core.models.episodes import EpisodeSpec
from mirrorbench.core.models.messages import Message, Role
from mirrorbench.core.models.plan import DatasetSpec, UserProxySpec
from mirrorbench.core.models.registry import ModelClientInfo
from mirrorbench.core.plan import Planner
from mirrorbench.core.registry.decorators import register_model_client
from mirrorbench.io.paths import Paths
from mirrorbench.model_clients.base import BaseChatClient, ChatChunk, ChatResponse
from mirrorbench.tasks.mirror_conversation import MIRROR_TASK_DRIVER_NAME, MirrorConversationDriver

SEEDED_ASSISTANT_TELEMETRY_COUNT = 1
GENERATED_ASSISTANT_TELEMETRY_COUNT = 2


@register_model_client(
    name="client:test/chat",
    metadata=ModelClientInfo(name="client:test/chat", provider="test", capabilities={"chat"}),
)
class StubChatClient(BaseChatClient):
    info: ClassVar[ModelClientInfo] = ModelClientInfo(
        name="client:test/chat",
        provider="test",
        capabilities={"chat"},
    )

    def __init__(self, respond_role: str = "assistant", fixed_response: str | None = None) -> None:
        self._respond_role = respond_role
        self._fixed_response = fixed_response

    def invoke(self, *, messages, **kwargs):  # type: ignore[override]
        content = self._fixed_response or f"{self._respond_role}:{len(messages)}"
        message = Message(role=Role(self._respond_role), content=content)
        usage = {
            "tokens_input": 1,
            "tokens_output": 1,
            "total_response_time": 0.1,
        }
        return ChatResponse(message=message, raw=None, usage=usage)

    def stream(self, *, messages, **kwargs):  # type: ignore[override]
        yield ChatChunk(delta=Message(role=Role(self._respond_role), content=""))


def test_planner_resolves_task_driver() -> None:
    job_cfg = JobConfig.model_validate(
        {
            "run": {},
            "user_proxies": [
                {
                    "name": "proxy:test",
                    "adapter": "adapter:generic/llm",
                    "params": {
                        "model_client": "client:test/chat",
                        "client_params": {"respond_role": "user", "fixed_response": "stub"},
                    },
                }
            ],
            "datasets": [
                {
                    "name": "dataset:jsonl/qulac_mirror",
                    "params": {},
                }
            ],
            "metrics": [
                {
                    "name": "metric:lexical/ttr",
                }
            ],
            "task_drivers": {
                "dataset:jsonl/qulac_mirror": {
                    "driver": MIRROR_TASK_DRIVER_NAME,
                    "params": {
                        "assistant_model_client": "client:test/chat",
                        "client_params": {
                            "respond_role": "assistant",
                            "fixed_response": "assistant reply",
                        },
                    },
                }
            },
        }
    )

    planner = Planner.from_config(job_cfg)
    planner.build()
    assert planner.manifest is not None
    driver_spec = planner.manifest.task_drivers["dataset:jsonl/qulac_mirror"]
    assert driver_spec.name == MIRROR_TASK_DRIVER_NAME
    assert driver_spec.params["assistant_model_client"] == "client:test/chat"


@pytest.mark.parametrize("starts_with_role", [Role.ASSISTANT, Role.USER])
def test_mirror_conversation_driver_generates_turns(tmp_path, starts_with_role):
    paths = Paths(tmp_path / "mirrorbench")
    run_config = RunConfig.model_validate({})

    driver = MirrorConversationDriver()
    driver.setup(
        run_id="run-1",
        run_config=run_config,
        paths=paths,
        dataset=DatasetSpec(name="dataset:test/mirror", task="mirror_conversation"),
        params={
            "assistant_model_client": "client:test/chat",
            "client_params": {"respond_role": "assistant", "fixed_response": "assistant reply"},
        },
    )

    adapter = LLMAdapter()
    adapter.configure_cache(cache_config=CacheConfig(enabled=False), paths=paths)
    session = adapter.spawn(
        config=UserProxySpec(
            name="proxy:test",
            params={
                "model_client": "client:test/chat",
                "client_params": {"respond_role": "user", "fixed_response": "user reply"},
            },
        ),
        run_id="run-1",
    )

    system_message = Message(role=Role.SYSTEM, content="system")
    initial_turn = Message(role=starts_with_role, content="initial")
    second_turn_role = Role.USER if starts_with_role is Role.ASSISTANT else Role.ASSISTANT
    second_turn = Message(role=second_turn_role, content="follow-up")
    episode = EpisodeSpec(
        episode_id="ep-1",
        task_tag="mirror_conversation",
        chat_history=[system_message, initial_turn, second_turn],
        references={"real_conversation": [initial_turn, second_turn]},
        metadata={},
    )

    result = driver.run_episode(episode=episode, proxy_session=session, run_id="run-1")
    turns = result.artifact.turns

    if starts_with_role is Role.ASSISTANT:
        assert [turn.role for turn in turns] == [Role.ASSISTANT, Role.USER]
    else:
        assert [turn.role for turn in turns] == [Role.USER, Role.ASSISTANT]
    assert any(turn.content == "user reply" for turn in turns)

    if starts_with_role is Role.ASSISTANT:
        assert turns[0].content == "initial"
        assert turns[1].content == "user reply"
        assert not any(turn.content == "assistant reply" for turn in turns)
        assert len(result.turn_telemetries) == SEEDED_ASSISTANT_TELEMETRY_COUNT
    else:
        assert any(turn.content == "assistant reply" for turn in turns)
        assert len(result.turn_telemetries) == GENERATED_ASSISTANT_TELEMETRY_COUNT

    session.shutdown()
    driver.shutdown()


@pytest.mark.parametrize("starts_with_role", [Role.ASSISTANT, Role.USER])
def test_mirror_conversation_driver_generates_turns_async(tmp_path, starts_with_role):
    paths = Paths(tmp_path / "mirrorbench")
    run_config = RunConfig.model_validate({})

    driver = MirrorConversationDriver()
    driver.setup(
        run_id="run-1",
        run_config=run_config,
        paths=paths,
        dataset=DatasetSpec(name="dataset:test/mirror", task="mirror_conversation"),
        params={
            "assistant_model_client": "client:test/chat",
            "client_params": {"respond_role": "assistant", "fixed_response": "assistant reply"},
        },
    )

    adapter = LLMAdapter()
    adapter.configure_cache(cache_config=CacheConfig(enabled=False), paths=paths)
    session = adapter.spawn(
        config=UserProxySpec(
            name="proxy:test",
            params={
                "model_client": "client:test/chat",
                "client_params": {"respond_role": "user", "fixed_response": "user reply"},
            },
        ),
        run_id="run-1",
    )

    system_message = Message(role=Role.SYSTEM, content="system")
    initial_turn = Message(role=starts_with_role, content="initial")
    second_turn_role = Role.USER if starts_with_role is Role.ASSISTANT else Role.ASSISTANT
    second_turn = Message(role=second_turn_role, content="follow-up")
    episode = EpisodeSpec(
        episode_id="ep-1",
        task_tag="mirror_conversation",
        chat_history=[system_message, initial_turn, second_turn],
        references={"real_conversation": [initial_turn, second_turn]},
        metadata={},
    )

    result = asyncio.run(
        driver.run_episode_async(episode=episode, proxy_session=session, run_id="run-1")
    )
    turns = result.artifact.turns

    if starts_with_role is Role.ASSISTANT:
        assert [turn.role for turn in turns] == [Role.ASSISTANT, Role.USER]
    else:
        assert [turn.role for turn in turns] == [Role.USER, Role.ASSISTANT]
    assert any(turn.content == "user reply" for turn in turns)

    if starts_with_role is Role.ASSISTANT:
        assert turns[0].content == "initial"
        assert turns[1].content == "user reply"
        assert not any(turn.content == "assistant reply" for turn in turns)
        assert len(result.turn_telemetries) == SEEDED_ASSISTANT_TELEMETRY_COUNT
    else:
        assert any(turn.content == "assistant reply" for turn in turns)
        assert len(result.turn_telemetries) == GENERATED_ASSISTANT_TELEMETRY_COUNT

    session.shutdown()
    driver.shutdown()
