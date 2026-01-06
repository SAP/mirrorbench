from __future__ import annotations

import asyncio
import importlib
import importlib.util
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest
from dotenv import load_dotenv

from mirrorbench.cache.manager import CacheManager
from mirrorbench.cache.sqlite_backend import SqliteCacheBackend
from mirrorbench.core.config import CacheConfig
from mirrorbench.core.constants import DEFAULT_CACHE_TTL_SECONDS
from mirrorbench.core.models.messages import Message, Role
from mirrorbench.core.models.registry import ModelClientInfo
from mirrorbench.model_clients.base import ChatClient, ChatResponse
from mirrorbench.model_clients.caching_wrapper import CachingChatClient
from mirrorbench.model_clients.langchain.chat import LangChainChatClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


EXPECTED_DELEGATE_CALLS = 2


class _FakeChatClient(ChatClient):
    info: ClassVar[ModelClientInfo] = ModelClientInfo(
        name="client:test",
        provider="test-provider",
        capabilities={"chat"},
    )

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def invoke(self, *, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        self.calls.append(list(messages))
        reply = Message(role=Role.ASSISTANT, content=f"reply-{len(self.calls)}")
        usage = {"tokens_input": 1, "tokens_output": 2}
        return ChatResponse(message=reply, usage=usage, raw={"echo": True})

    def stream(
        self, *, messages: Sequence[Message], **kwargs: Any
    ) -> Iterator[Any]:  # pragma: no cover - passthrough helper
        yield from ()

    def get_init_params(self) -> dict[str, Any]:
        return {"model": "test-model"}


class _AsyncChatClient(ChatClient):
    info: ClassVar[ModelClientInfo] = ModelClientInfo(
        name="client:test/async",
        provider="test-provider",
        capabilities={"chat"},
    )

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []
        self.async_calls = 0

    def invoke(self, *, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        self.calls.append(list(messages))
        reply = Message(role=Role.ASSISTANT, content=f"reply-{len(self.calls)}")
        usage = {"tokens_input": 1, "tokens_output": 2}
        return ChatResponse(message=reply, usage=usage, raw={"echo": True})

    async def invoke_async(self, *, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        self.async_calls += 1
        return self.invoke(messages=messages, **kwargs)

    def stream(
        self, *, messages: Sequence[Message], **kwargs: Any
    ) -> Iterator[Any]:  # pragma: no cover - passthrough helper
        yield from ()

    def get_init_params(self) -> dict[str, Any]:
        return {"model": "test-model"}


class _CountingChatClient(ChatClient):
    def __init__(self, delegate: ChatClient) -> None:
        self._delegate = delegate
        self.info = delegate.info
        self.calls = 0

    def invoke(self, *, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        self.calls += 1
        return self._delegate.invoke(messages=messages, **kwargs)

    def stream(
        self, *, messages: Sequence[Message], **kwargs: Any
    ):  # pragma: no cover - passthrough
        yield from self._delegate.stream(messages=messages, **kwargs)

    def get_init_params(self) -> dict[str, Any]:
        if hasattr(self._delegate, "get_init_params"):
            return self._delegate.get_init_params()
        return {}


def _build_manager(tmp_path) -> CacheManager:
    backend = SqliteCacheBackend(tmp_path / "cache.db")
    config = CacheConfig(enabled=True, ttl_seconds=None)
    return CacheManager(backend=backend, config=config)


def _has_azure_env() -> bool:
    return bool(os.getenv("AZURE_OPENAI_API_KEY")) and bool(os.getenv("AZURE_OPENAI_ENDPOINT"))


def test_caching_chat_client_returns_cached_response(tmp_path) -> None:
    manager = _build_manager(tmp_path)
    delegate = _FakeChatClient()
    client = CachingChatClient(
        delegate=delegate,
        manager=manager,
        namespace="model_client:client:test",
        ttl_seconds=None,
    )
    messages = [Message(role=Role.USER, content="hello")]

    first = client.invoke(messages=messages)
    second = client.invoke(messages=messages)

    # underlying client invoked only once
    assert len(delegate.calls) == 1
    assert first.message.content == "reply-1"
    assert second.message.content == "reply-1"
    assert first.usage["cache_hit"] is False
    assert second.usage["cache_hit"] is True


def test_caching_chat_client_invoke_async_returns_cached_response(tmp_path) -> None:
    manager = _build_manager(tmp_path)
    delegate = _AsyncChatClient()
    client = CachingChatClient(
        delegate=delegate,
        manager=manager,
        namespace="model_client:client:test/async",
        ttl_seconds=None,
    )
    messages = [Message(role=Role.USER, content="hello")]

    first = asyncio.run(client.invoke_async(messages=messages))
    second = asyncio.run(client.invoke_async(messages=messages))

    assert delegate.async_calls == 1
    assert first.message.content == "reply-1"
    assert second.message.content == "reply-1"
    assert first.usage["cache_hit"] is False
    assert second.usage["cache_hit"] is True


def test_disabled_manager_bypasses_cache(tmp_path) -> None:
    backend = SqliteCacheBackend(tmp_path / "cache.db")
    config = CacheConfig(enabled=False)
    manager = CacheManager(backend=backend, config=config)
    delegate = _FakeChatClient()
    client = CachingChatClient(
        delegate=delegate,
        manager=manager,
        namespace="model_client:client:test",
        ttl_seconds=None,
    )
    messages = [Message(role=Role.USER, content="hello")]

    client.invoke(messages=messages)
    client.invoke(messages=messages)

    # both calls hit the delegate when cache disabled
    assert len(delegate.calls) == EXPECTED_DELEGATE_CALLS


def test_different_init_params_produce_different_cache_keys(tmp_path) -> None:
    """Verify that clients with different init params don't share cache entries."""
    manager = _build_manager(tmp_path)
    messages = [Message(role=Role.USER, content="hello")]

    # Create two fake clients with different init params
    class _FakeChatClientWithModel(ChatClient):
        info: ClassVar[ModelClientInfo] = ModelClientInfo(
            name="client:test",
            provider="test-provider",
            capabilities={"chat"},
        )

        def __init__(self, model: str) -> None:
            self._model = model
            self.calls: list[list[Message]] = []

        def invoke(self, *, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
            self.calls.append(list(messages))
            reply = Message(role=Role.ASSISTANT, content=f"reply-{self._model}-{len(self.calls)}")
            usage = {"tokens_input": 1, "tokens_output": 2}
            return ChatResponse(message=reply, usage=usage, raw={"echo": True})

        def stream(self, *, messages: Sequence[Message], **kwargs: Any) -> Iterator[Any]:
            yield from ()

        def get_init_params(self) -> dict[str, Any]:
            return {"model": self._model}

    delegate1 = _FakeChatClientWithModel("gpt-4o")
    client1 = CachingChatClient(
        delegate=delegate1,
        manager=manager,
        namespace="model_client:client:test",
        ttl_seconds=None,
    )

    delegate2 = _FakeChatClientWithModel("gpt-4.1")
    client2 = CachingChatClient(
        delegate=delegate2,
        manager=manager,
        namespace="model_client:client:test",
        ttl_seconds=None,
    )

    # Both clients should invoke their delegates despite same messages
    resp1 = client1.invoke(messages=messages)
    resp2 = client2.invoke(messages=messages)

    # Each delegate should be called once (no cache sharing)
    assert len(delegate1.calls) == 1
    assert len(delegate2.calls) == 1

    # Responses should be different
    assert resp1.message.content == "reply-gpt-4o-1"
    assert resp2.message.content == "reply-gpt-4.1-1"

    # Cache hits should be false for both
    assert resp1.usage["cache_hit"] is False
    assert resp2.usage["cache_hit"] is False

    # Repeated calls should hit cache for each client separately
    resp1_cached = client1.invoke(messages=messages)
    resp2_cached = client2.invoke(messages=messages)

    assert len(delegate1.calls) == 1  # Still only one call
    assert len(delegate2.calls) == 1  # Still only one call
    assert resp1_cached.usage["cache_hit"] is True
    assert resp2_cached.usage["cache_hit"] is True
    assert resp1_cached.message.content == "reply-gpt-4o-1"
    assert resp2_cached.message.content == "reply-gpt-4.1-1"


@pytest.mark.real_azure_langchain
@pytest.mark.skipif(not _has_azure_env(), reason="Azure OpenAI credentials not configured")
def test_caching_wrapper_with_real_azure_langchain(tmp_path) -> None:
    if importlib.util.find_spec("langchain_openai") is None:
        pytest.skip("langchain-openai package not installed")

    module = importlib.import_module("langchain_openai")
    azure_chat_openai_cls = module.AzureChatOpenAI

    azure_model = azure_chat_openai_cls(
        azure_deployment="gpt-4o",
        api_version="2025-01-01-preview",
    )

    langchain_client = LangChainChatClient(chat_model=azure_model)
    counting_client = _CountingChatClient(langchain_client)

    manager = CacheManager(
        backend=SqliteCacheBackend(tmp_path / "cache.db"),
        config=CacheConfig(
            enabled=True,
            ttl_seconds=DEFAULT_CACHE_TTL_SECONDS,
            backend="sqlite",
        ),
    )

    client = CachingChatClient(
        delegate=counting_client,
        manager=manager,
        namespace="model_client:azure",
        ttl_seconds=None,
    )

    messages = [Message(role=Role.USER, content="Reply with pong.")]

    first = client.invoke(messages=messages)
    second = client.invoke(messages=messages)

    assert "pong" in first.message.content.lower()
    assert second.message.content == first.message.content
    assert counting_client.calls == 1
    stats = manager.stats(namespace="model_client:azure")
    assert stats and stats[0].entries == 1
