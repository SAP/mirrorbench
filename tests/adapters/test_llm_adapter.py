from __future__ import annotations

import asyncio
import importlib
import importlib.util
import os
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from dotenv import load_dotenv

import mirrorbench.adapters  # noqa: F401 - ensure registration side effects
from mirrorbench.adapters.llm import GenerationResult, LLMAdapter
from mirrorbench.core.config import CacheConfig
from mirrorbench.core.models.episodes import EpisodeSpec
from mirrorbench.core.models.messages import Message, Role
from mirrorbench.core.models.plan import UserProxySpec
from mirrorbench.core.models.registry import ModelClientInfo
from mirrorbench.core.registry import registry
from mirrorbench.core.registry.decorators import register_model_client
from mirrorbench.io.paths import Paths
from mirrorbench.model_clients.base import BaseChatClient, ChatChunk, ChatResponse
from mirrorbench.model_clients.caching_wrapper import CachingChatClient

ADAPTER_NAME = "adapter:generic/llm"

STUB_CLIENT_INFO = ModelClientInfo(
    name="client:test/llm_adapter",
    provider="unit-test",
    capabilities={"chat", "streaming"},
)
ASYNC_STUB_CLIENT_INFO = ModelClientInfo(
    name="client:test/llm_adapter_async",
    provider="unit-test",
    capabilities={"chat", "streaming"},
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


@register_model_client(
    name=STUB_CLIENT_INFO.name,
    metadata=STUB_CLIENT_INFO,
)
class _StubChatClient(BaseChatClient):
    info = STUB_CLIENT_INFO

    def __init__(self, *, prefix: str = "stub", enable_stream: bool = False) -> None:
        self.prefix = prefix
        self.enable_stream = enable_stream
        self.invocations: list[tuple[list[Message], dict[str, object]]] = []
        self.stream_invocations: list[tuple[list[Message], dict[str, object]]] = []
        self.closed = False

    def invoke(self, *, messages: Sequence[Message], **kwargs: object) -> ChatResponse:  # type: ignore[override]
        self.invocations.append((list(messages), dict(kwargs)))
        last = messages[-1]
        content = f"{self.prefix}:{last.content}"
        usage = {
            "tokens_input": 5,
            "tokens_output": 7,
            "total_response_time": 0.42,
        }
        return ChatResponse(
            message=Message(role=Role.ASSISTANT, content=content),
            raw={"echo": content},
            usage=usage,
        )

    def stream(self, *, messages: Sequence[Message], **kwargs: object) -> Iterator[ChatChunk]:  # type: ignore[override]
        self.stream_invocations.append((list(messages), dict(kwargs)))
        if not self.enable_stream:
            yield from super().stream(messages=messages, **kwargs)
            return
        yield ChatChunk(delta=Message(role=Role.ASSISTANT, content=f"{self.prefix}:chunk1"))
        yield ChatChunk(delta=Message(role=Role.ASSISTANT, content=f"{self.prefix}:chunk2"))

    def shutdown(self) -> None:  # - simple flag setter
        self.closed = True


@register_model_client(
    name=ASYNC_STUB_CLIENT_INFO.name,
    metadata=ASYNC_STUB_CLIENT_INFO,
)
class _AsyncStubChatClient(BaseChatClient):
    info = ASYNC_STUB_CLIENT_INFO

    def __init__(self, *, prefix: str = "async") -> None:
        self.prefix = prefix
        self.async_calls = 0
        self.sync_calls = 0

    def invoke(self, *, messages: Sequence[Message], **kwargs: object) -> ChatResponse:  # type: ignore[override]
        self.sync_calls += 1
        last = messages[-1]
        content = f"{self.prefix}:{last.content}"
        return ChatResponse(message=Message(role=Role.ASSISTANT, content=content))

    async def invoke_async(self, *, messages: Sequence[Message], **kwargs: object) -> ChatResponse:
        self.async_calls += 1
        last = messages[-1]
        content = f"{self.prefix}:{last.content}"
        return ChatResponse(message=Message(role=Role.ASSISTANT, content=content))


def _build_episode() -> EpisodeSpec:
    history = [Message(role=Role.USER, content="Hello there!")]
    return EpisodeSpec(
        episode_id="ep-1",
        task_tag="unit",
        chat_history=history,
    )


def test_llm_adapter_is_registered() -> None:
    entry = registry.get("user_proxies", ADAPTER_NAME)
    assert entry.metadata is not None
    factory = registry.factory("user_proxies", ADAPTER_NAME)
    adapter = factory()
    assert isinstance(adapter, LLMAdapter)


def test_llm_session_invoke_uses_model_client_defaults() -> None:
    adapter_factory = registry.factory("user_proxies", ADAPTER_NAME)
    adapter = adapter_factory()
    adapter.configure_cache(cache_config=CacheConfig(enabled=False), paths=Paths.default())

    spec = UserProxySpec(
        name="unit-proxy",
        adapter=ADAPTER_NAME,
        params={
            "model_client": STUB_CLIENT_INFO.name,
            "client_params": {"prefix": "answer"},
            "request_params": {"temperature": 0.3},
            "system_prompt": "You are a helpful test stub.",
        },
    )
    session = adapter.spawn(config=spec, run_id="run-001")

    result = session.generate(turn=_build_episode())
    assert isinstance(result, GenerationResult)
    assert result.message.content == "answer:Hello there!"
    assert result.telemetry.tokens_input == 5  # noqa: PLR2004
    assert result.telemetry.tokens_output == 7  # noqa: PLR2004
    assert result.telemetry.total_response_time == pytest.approx(0.42)
    assert result.telemetry.provider == STUB_CLIENT_INFO.provider
    assert result.raw == {"echo": "answer:Hello there!"}

    client = session.client
    if isinstance(client, CachingChatClient):
        client = client._delegate  # type: ignore[attr-defined]
    assert isinstance(client, _StubChatClient)
    recorded_messages, recorded_kwargs = client.invocations[-1]
    assert recorded_messages[0].role is Role.SYSTEM
    assert recorded_messages[0].content == "You are a helpful test stub."
    assert recorded_messages[1].content == "Hello there!"
    assert recorded_kwargs == {"temperature": 0.3}

    session.shutdown()
    assert client.closed is True


def test_llm_session_generate_async_prefers_async_client() -> None:
    adapter_factory = registry.factory("user_proxies", ADAPTER_NAME)
    adapter = adapter_factory()
    adapter.configure_cache(cache_config=CacheConfig(enabled=False), paths=Paths.default())

    spec = UserProxySpec(
        name="async-proxy",
        adapter=ADAPTER_NAME,
        params={
            "model_client": ASYNC_STUB_CLIENT_INFO.name,
            "client_params": {"prefix": "async"},
        },
    )
    session = adapter.spawn(config=spec, run_id="run-async")

    result = asyncio.run(session.generate_async(turn=_build_episode()))
    assert result.message.content == "async:Hello there!"

    client = session.client
    assert isinstance(client, _AsyncStubChatClient)
    assert client.async_calls == 1
    assert client.sync_calls == 0


def test_llm_session_streams_chunks_when_supported() -> None:
    adapter = registry.factory("user_proxies", ADAPTER_NAME)()
    spec = UserProxySpec(
        name="stream-proxy",
        adapter=ADAPTER_NAME,
        params={
            "model_client": STUB_CLIENT_INFO.name,
            "client_params": {"prefix": "stream", "enable_stream": True},
        },
    )
    session = adapter.spawn(config=spec, run_id="run-stream")

    chunks = list(session.stream(turn=_build_episode()))
    assert [chunk.delta.content for chunk in chunks if chunk.delta] == [
        "stream:chunk1",
        "stream:chunk2",
    ]


def test_llm_adapter_requires_model_client_name() -> None:
    adapter = registry.factory("user_proxies", ADAPTER_NAME)()
    spec = UserProxySpec(name="missing-client", adapter=ADAPTER_NAME, params={})
    with pytest.raises(ValueError):
        adapter.spawn(config=spec, run_id="run-missing")


def test_llm_adapter_validates_client_params_type() -> None:
    adapter = registry.factory("user_proxies", ADAPTER_NAME)()
    spec = UserProxySpec(
        name="bad-client-params",
        adapter=ADAPTER_NAME,
        params={
            "model_client": STUB_CLIENT_INFO.name,
            "client_params": ["not-a-mapping"],
        },
    )
    with pytest.raises(TypeError):
        adapter.spawn(config=spec, run_id="run-bad-client")


def _has_azure_env() -> bool:
    return bool(os.getenv("AZURE_OPENAI_API_KEY")) and bool(os.getenv("AZURE_OPENAI_ENDPOINT"))


@pytest.mark.real_azure_langchain
@pytest.mark.skipif(not _has_azure_env(), reason="Azure OpenAI credentials not configured")
def test_llm_adapter_with_langchain_azure() -> None:
    if importlib.util.find_spec("langchain_openai") is None:
        pytest.skip("langchain-openai package not installed")
    module = importlib.import_module("langchain_openai")
    azure_chat_openai_cls = module.AzureChatOpenAI

    adapter = registry.factory("user_proxies", ADAPTER_NAME)()
    proxy_spec = UserProxySpec(
        name="azure-langchain",
        adapter=ADAPTER_NAME,
        params={
            "model_client": "client:langchain/chat",
            "client_params": {
                "chat_model": azure_chat_openai_cls(
                    azure_deployment="gpt-4o",
                    api_version="2025-01-01-preview",
                )
            },
            "system_prompt": "You are an evaluation assistant; respond with a concise pong.",
        },
    )
    session = adapter.spawn(config=proxy_spec, run_id="run-azure")
    episode = EpisodeSpec(
        episode_id="azure-ep",
        task_tag="chat",
        chat_history=[Message(role=Role.USER, content="Reply with pong.")],
    )

    result = session.generate(turn=episode)
    assert "pong" in result.message.content.lower()
    session.shutdown()


def test_llm_adapter_tags_telemetry_with_component() -> None:
    """Test that user proxy telemetry is tagged with component='user_proxy'."""
    adapter = registry.factory("user_proxies", ADAPTER_NAME)()
    adapter.configure_cache(cache_config=CacheConfig(enabled=False), paths=Paths.default())

    spec = UserProxySpec(
        name="telemetry-test-proxy",
        adapter=ADAPTER_NAME,
        params={
            "model_client": STUB_CLIENT_INFO.name,
            "client_params": {"prefix": "test"},
        },
    )
    session = adapter.spawn(config=spec, run_id="run-telemetry-test")

    result = session.generate(turn=_build_episode())

    # Verify telemetry has component tag
    assert result.telemetry is not None
    assert result.telemetry.metadata is not None
    assert "component" in result.telemetry.metadata
    assert result.telemetry.metadata["component"] == "user_proxy"

    # Verify other telemetry fields are still present
    assert result.telemetry.tokens_input == 5  # noqa: PLR2004
    assert result.telemetry.tokens_output == 7  # noqa: PLR2004
    assert result.telemetry.provider == STUB_CLIENT_INFO.provider


def test_llm_adapter_includes_rich_proxy_metadata_in_telemetry() -> None:
    """Test that telemetry includes rich proxy specification details."""

    # Create a custom stub client that accepts additional parameters
    @register_model_client(
        name="client:test/rich-telemetry",
        metadata=ModelClientInfo(
            name="client:test/rich-telemetry",
            provider="unit-test-rich",
            capabilities={"chat"},
        ),
    )
    class RichTelemetryStubClient(BaseChatClient):
        info = ModelClientInfo(
            name="client:test/rich-telemetry",
            provider="unit-test-rich",
            capabilities={"chat"},
        )

        def __init__(
            self, *, prefix: str = "stub", model: str | None = None, model_name: str | None = None
        ) -> None:
            self.prefix = prefix
            self.model = model
            self.model_name = model_name

        def invoke(self, *, messages: Sequence[Message], **kwargs: object) -> ChatResponse:  # type: ignore[override]
            usage = {
                "tokens_input": 10,
                "tokens_output": 15,
                "total_response_time": 0.5,
            }
            return ChatResponse(
                message=Message(role=Role.ASSISTANT, content=f"{self.prefix}:response"),
                raw={},
                usage=usage,
            )

    adapter = registry.factory("user_proxies", ADAPTER_NAME)()
    adapter.configure_cache(cache_config=CacheConfig(enabled=False), paths=Paths.default())

    spec = UserProxySpec(
        name="rich-telemetry-proxy",
        adapter=ADAPTER_NAME,
        params={
            "model_client": "client:test/rich-telemetry",
            "client_params": {
                "prefix": "test",
                "model": "gpt-4o",
                "model_name": "gpt-4o-2024-05-13",
            },
            "request_params": {
                "temperature": 0.7,
                "max_tokens": 1000,
            },
        },
    )
    session = adapter.spawn(config=spec, run_id="run-rich-telemetry")

    result = session.generate(turn=_build_episode())

    # Verify rich telemetry metadata
    assert result.telemetry is not None
    assert result.telemetry.metadata is not None

    # Verify proxy specification details
    assert result.telemetry.metadata["proxy_name"] == "rich-telemetry-proxy"
    assert result.telemetry.metadata["proxy_adapter"] == ADAPTER_NAME
    assert result.telemetry.metadata["model_client"] == "client:test/rich-telemetry"

    # Verify model parameters
    assert result.telemetry.metadata["model"] == "gpt-4o"
    assert result.telemetry.metadata["model_name"] == "gpt-4o-2024-05-13"

    # Verify request parameters
    assert result.telemetry.metadata["temperature"] == 0.7  # noqa: PLR2004
    assert result.telemetry.metadata["max_tokens"] == 1000  # noqa: PLR2004
