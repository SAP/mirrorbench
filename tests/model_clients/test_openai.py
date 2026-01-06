from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from dotenv import load_dotenv
from pydantic import SecretStr

from mirrorbench.core.constants import OPENAI_API_KEY_ENV, OPENAI_ORG_ID_ENV
from mirrorbench.core.models.messages import Message, Role
from mirrorbench.model_clients.exceptions import ModelClientError
from mirrorbench.model_clients.openai.chat import OpenAIChatClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


class _FakeOpenAIStream(list):
    pass


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_args: dict[str, object] | None = None

    def create(self, *, model: str, messages, stream: bool, **kwargs) -> object:
        self.last_args = {"model": model, "messages": messages, "stream": stream, **kwargs}
        if stream:
            return _FakeOpenAIStream(
                [
                    {"choices": [{"delta": {"content": "part1"}}]},
                    {"choices": [{"delta": {"content": "part2"}}]},
                ]
            )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hello",
                    }
                }
            ],
            "usage": {"tokens_input": 1, "tokens_output": 1},
        }


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeOpenAI:
    def __init__(self, *, api_key: str | None = None, organization: str | None = None) -> None:
        self.api_key = api_key
        self.organization = organization
        self.chat = _FakeChat()


@pytest.fixture(autouse=True)
def _patch_openai(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("real_openai") is not None:
        yield
        return

    module = SimpleNamespace(OpenAI=_FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv(OPENAI_API_KEY_ENV, "secret")
    monkeypatch.delenv(OPENAI_ORG_ID_ENV, raising=False)
    yield
    sys.modules.pop("openai", None)


def test_openai_chat_client_invoke() -> None:
    client = OpenAIChatClient()
    response = client.invoke(messages=[], temperature=0.4)
    assert isinstance(client._api_key, SecretStr)
    assert response.message.content == "hello"
    assert response.usage.get("provider") == "openai"
    assert "total_response_time" in response.usage
    assert client._client.chat.completions.last_args["temperature"] == 0.4  # noqa: PLR2004


def test_openai_chat_client_invoke_stream_flag() -> None:
    client = OpenAIChatClient()
    response = client.invoke(messages=[], stream=True, temperature=0.2)
    assert response.message.content == "part1part2"
    assert response.usage is not None
    assert response.usage.get("provider") == "openai"
    assert client._client.chat.completions.last_args["temperature"] == 0.2  # noqa: PLR2004


def test_openai_chat_client_stream() -> None:
    client = OpenAIChatClient()
    chunks = list(client.stream(messages=[], temperature=0.1))
    assert [chunk.delta.content for chunk in chunks] == ["part1", "part2"]
    for chunk in chunks:
        assert chunk.telemetry is not None
    assert client._client.chat.completions.last_args["temperature"] == 0.1  # noqa: PLR2004


def test_openai_chat_client_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OPENAI_API_KEY_ENV, raising=False)
    sys.modules.pop("openai", None)
    client = OpenAIChatClient()
    with pytest.raises(ModelClientError):
        client.invoke(messages=[])


@pytest.mark.real_openai
@pytest.mark.skipif(not os.getenv(OPENAI_API_KEY_ENV), reason="OpenAI API key not configured")
def test_openai_chat_client_real_invoke_stream() -> None:
    if importlib.util.find_spec("openai") is None:
        pytest.skip("openai package not installed")

    client = OpenAIChatClient()
    response = client.invoke(
        messages=[Message(role=Role.USER, content="Respond with the word 'pong'.")],
        stream=True,
    )

    assert response.message.content
    assert "pong" in response.message.content.lower()
    assert response.usage is not None
    assert response.usage.get("provider") == "openai"
