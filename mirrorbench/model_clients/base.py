"""Base interfaces and helpers for model clients."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from mirrorbench.core.models.messages import Message
from mirrorbench.core.models.registry import ModelClientInfo
from mirrorbench.core.registry.components import BaseModelClient


@dataclass(slots=True)
class ChatResponse:
    """Normalized response returned by chat model clients."""

    message: Message
    raw: Mapping[str, Any] | None = None
    usage: Mapping[str, Any] | None = None


@dataclass(slots=True)
class ChatChunk:
    """Incremental payload emitted during streaming responses."""

    delta: Message | None = None
    raw: Mapping[str, Any] | None = None
    telemetry: dict[str, Any] | None = None


class ChatClient(Protocol):
    """Protocol implemented by chat-centric model clients."""

    info: ClassVar[ModelClientInfo]

    def invoke(self, *, messages: Sequence[Message], **kwargs: Any) -> ChatResponse: ...

    async def invoke_async(self, *, messages: Sequence[Message], **kwargs: Any) -> ChatResponse: ...

    def stream(self, *, messages: Sequence[Message], **kwargs: Any) -> Iterator[ChatChunk]: ...

    def get_init_params(self) -> Mapping[str, Any]:
        """Return initialization parameters that affect model behavior.

        These parameters are used in cache key generation to ensure that
        different model configurations produce different cache keys.

        Returns:
            A mapping of parameter names to values that uniquely identify
            the model configuration (e.g., model name, temperature defaults).
        """
        ...


class BaseChatClient(BaseModelClient, ChatClient):
    """Convenience base providing default streaming behaviour."""

    info: ClassVar[ModelClientInfo]

    def stream(self, *, messages: Sequence[Message], **kwargs: Any) -> Iterator[ChatChunk]:
        response = self.invoke(messages=messages, **kwargs)
        telemetry = dict(response.usage or {})
        telemetry.setdefault("time_to_first_token", None)
        telemetry.setdefault("time_per_output_token", None)
        chunk = ChatChunk(delta=response.message, raw=response.raw, telemetry=telemetry)
        yield chunk

    async def invoke_async(self, *, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        return await asyncio.to_thread(self.invoke, messages=messages, **kwargs)

    def get_init_params(self) -> Mapping[str, Any]:
        """Return initialization parameters for cache key generation.

        Default implementation returns an empty dict. Subclasses should
        override to include parameters that affect model behavior.
        """
        return {}


__all__ = [
    "BaseChatClient",
    "ChatChunk",
    "ChatClient",
    "ChatResponse",
]
