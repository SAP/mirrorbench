from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from dotenv import load_dotenv

from mirrorbench.core.constants import OPENAI_API_KEY_ENV
from mirrorbench.core.models.messages import Message, Role
from mirrorbench.model_clients import telemetry as telemetry_mod
from mirrorbench.model_clients.exceptions import AuthenticationError, ModelClientError
from mirrorbench.model_clients.langchain import chat as chat_module
from mirrorbench.model_clients.langchain.chat import LangChainChatClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


class _DummyLangChainModel:
    def __init__(self, *, streaming: bool = False) -> None:
        self.streaming = streaming
        self.last_kwargs: dict[str, object] | None = None

    def invoke(self, messages, **kwargs) -> object:
        self.last_kwargs = dict(kwargs)
        last = messages[-1]
        content = last.content if isinstance(last, Message) else getattr(last, "content", "")
        return type(
            "Result",
            (),
            {
                "content": f"echo:{content}",
                "usage_metadata": {"input_tokens": 1, "output_tokens": 2},
            },
        )()

    def stream(self, messages, **kwargs) -> Iterator[object]:
        if not self.streaming:
            raise AttributeError("streaming not supported")
        yield type("Result", (), {"content": "chunk1"})()
        yield type("Result", (), {"content": "chunk2"})()


class _FlakyLangChainModel(_DummyLangChainModel):
    def __init__(self, *, failures: int) -> None:
        super().__init__()
        self.failures = failures
        self.attempts = 0

    def invoke(self, messages, **kwargs) -> object:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise RuntimeError("temporary failure")
        return super().invoke(messages, **kwargs)


class _FakeOpenAIModel(_DummyLangChainModel):
    __module__ = "langchain_openai.ChatOpenAI._fake"


@pytest.fixture
def no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    def patched_invoke(info, call):
        if hasattr(call, "retry"):
            call.retry.sleep = lambda _: None  # type: ignore[attr-defined]
        return telemetry_mod.invoke_with_telemetry(info, call)

    monkeypatch.setattr(chat_module, "invoke_with_telemetry", patched_invoke)


def test_langchain_chat_invoke() -> None:
    client = LangChainChatClient(chat_model=_DummyLangChainModel())
    response = client.invoke(messages=[Message(role=Role.USER, content="hi")], temperature=0.2)
    assert response.message.content == "echo:hi"
    assert response.usage["tokens_input"] == 1
    assert response.usage["tokens_output"] == 2  # noqa: PLR2004
    assert response.usage["total_response_time"] >= 0
    assert client._chat_model.last_kwargs == {"temperature": 0.2}


def test_langchain_chat_invoke_retries_success(no_retry_sleep: None) -> None:
    model = _FlakyLangChainModel(failures=2)
    client = LangChainChatClient(chat_model=model)

    response = client.invoke(messages=[Message(role=Role.USER, content="hi")])

    assert response.message.content == "echo:hi"
    assert model.attempts == model.failures + 1


def test_langchain_chat_invoke_retries_exhaust(no_retry_sleep: None) -> None:
    model = _FlakyLangChainModel(failures=10)
    client = LangChainChatClient(chat_model=model)

    with pytest.raises(ModelClientError):
        client.invoke(messages=[Message(role=Role.USER, content="hi")])

    assert model.attempts == 5  # noqa: PLR2004


def test_langchain_chat_stream() -> None:
    client = LangChainChatClient(chat_model=_DummyLangChainModel(streaming=True))
    chunks = list(client.stream(messages=[Message(role=Role.USER, content="hi")]))
    assert [chunk.delta.content for chunk in chunks if chunk.delta] == ["chunk1", "chunk2"]
    assert chunks[-1].telemetry["total_response_time"] >= 0


def test_langchain_chat_factory_initialization() -> None:
    client = LangChainChatClient(
        model_import="tests.model_clients.test_langchain._DummyLangChainModel",
        model_kwargs={"streaming": True},
    )
    response = client.invoke(messages=[Message(role=Role.USER, content="hello")])
    assert response.message.content == "echo:hello"
    chunks = list(client.stream(messages=[Message(role=Role.USER, content="hi")]))
    assert [chunk.delta.content for chunk in chunks if chunk.delta] == ["chunk1", "chunk2"]


def _has_azure_env() -> bool:
    return bool(os.getenv("AZURE_OPENAI_API_KEY")) and bool(os.getenv("AZURE_OPENAI_ENDPOINT"))


@pytest.mark.real_azure_langchain
@pytest.mark.skipif(not _has_azure_env(), reason="Azure OpenAI credentials not configured")
def test_langchain_chat_with_azure_instance() -> None:
    if importlib.util.find_spec("langchain_openai") is None:
        pytest.skip("langchain-openai package not installed")
    module = importlib.import_module("langchain_openai")
    azure_chat_openai_cls = module.AzureChatOpenAI

    model = azure_chat_openai_cls(
        azure_deployment="gpt-4o",
        api_version="2025-01-01-preview",
    )
    client = LangChainChatClient(chat_model=model)
    response = client.invoke(messages=[Message(role=Role.USER, content="Reply with pong.")])
    assert "pong" in response.message.content.lower()


@pytest.mark.real_azure_langchain
@pytest.mark.skipif(not _has_azure_env(), reason="Azure OpenAI credentials not configured")
def test_langchain_chat_with_azure_import() -> None:
    if importlib.util.find_spec("langchain_openai") is None:
        pytest.skip("langchain-openai package not installed")

    client = LangChainChatClient(
        model_import="langchain_openai.AzureChatOpenAI",
        model_kwargs={
            "azure_deployment": "gpt-4o",
            "api_version": "2025-01-01-preview",
        },
    )
    response = client.invoke(messages=[Message(role=Role.USER, content="Reply with pong.")])
    assert "pong" in response.message.content.lower()


def test_langchain_chat_validate_credentials_with_dummy(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LangChainChatClient(chat_model=_DummyLangChainModel())
    client.validate_credentials()  # Should not raise for non-OpenAI models


def test_langchain_chat_validate_credentials_requires_openai_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OPENAI_API_KEY_ENV, raising=False)
    client = LangChainChatClient(chat_model=_FakeOpenAIModel())
    with pytest.raises(AuthenticationError):
        client.validate_credentials()

    monkeypatch.setenv(OPENAI_API_KEY_ENV, "test-key")
    client.validate_credentials()
