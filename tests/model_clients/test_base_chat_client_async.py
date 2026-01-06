from __future__ import annotations

import asyncio
from typing import ClassVar

from mirrorbench.core.models.messages import Message, Role
from mirrorbench.core.models.registry import ModelClientInfo
from mirrorbench.model_clients.base import BaseChatClient, ChatResponse


class _BaseAsyncClient(BaseChatClient):
    info: ClassVar[ModelClientInfo] = ModelClientInfo(
        name="client:test/base-async",
        provider="test-provider",
        capabilities={"chat"},
    )

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, *, messages, **kwargs):  # type: ignore[override]
        self.calls += 1
        last = messages[-1]
        return ChatResponse(message=Message(role=Role.ASSISTANT, content=f"ok:{last.content}"))


def test_base_chat_client_invoke_async_delegates_to_invoke() -> None:
    client = _BaseAsyncClient()
    response = asyncio.run(client.invoke_async(messages=[Message(role=Role.USER, content="hi")]))
    assert client.calls == 1
    assert response.message.content == "ok:hi"
