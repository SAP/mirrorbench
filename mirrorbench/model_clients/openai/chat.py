"""OpenAI chat client implementation."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, ClassVar, cast

from pydantic import SecretStr
from tenacity import retry, stop_after_attempt, wait_exponential

from mirrorbench.core.constants import OPENAI_API_KEY_ENV, OPENAI_ORG_ID_ENV
from mirrorbench.core.env import get_env
from mirrorbench.core.models.messages import Message, Role
from mirrorbench.core.models.registry import ModelClientInfo
from mirrorbench.core.registry.decorators import register_model_client
from mirrorbench.model_clients.base import BaseChatClient, ChatChunk, ChatResponse
from mirrorbench.model_clients.exceptions import AuthenticationError, ModelClientError
from mirrorbench.model_clients.telemetry import (
    invoke_with_telemetry,
    normalize_usage,
    stream_with_telemetry,
)
from mirrorbench.model_clients.utils import coerce_int

OPENAI_RETRY_DECORATOR = cast(
    Callable[[Callable[[], ChatResponse]], Callable[[], ChatResponse]],
    retry(
        wait=wait_exponential(multiplier=2, min=1, max=60), stop=stop_after_attempt(5), reraise=True
    ),
)

OPENAI_CHAT_INFO = ModelClientInfo(
    name="client:openai/chat",
    provider="openai",
    capabilities={"chat", "streaming"},
    models={"gpt-3.5-turbo", "gpt-4o"},
    telemetry_keys={
        "tokens_input",
        "tokens_output",
        "time_to_first_token",
        "time_per_output_token",
        "total_response_time",
    },
)


def _format_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for message in messages:
        payload = {"role": message.role.value, "content": message.content}
        tool_calls = message.metadata.get("tool_calls") if message.metadata else None
        if tool_calls:
            payload["tool_calls"] = tool_calls
        formatted.append(payload)
    return formatted


def _extract_message(choice: Any) -> Message:
    content = ""
    role = Role.ASSISTANT
    metadata: dict[str, Any] = {}
    try:
        message_dict = choice["message"]
        role_value = message_dict.get("role", "assistant")
        content = message_dict.get("content", "")
        role = Role(role_value)
        tool_calls = message_dict.get("tool_calls")
        if tool_calls:
            metadata["tool_calls"] = tool_calls
    except Exception:  # pragma: no cover - defensive fallback
        pass
    return Message(role=role, content=content, metadata=metadata)


def _convert_openai_usage(info: ModelClientInfo, usage: Mapping[str, Any] | None) -> dict[str, Any]:
    tokens: dict[str, Any] = {}
    if isinstance(usage, Mapping):
        prompt_tokens = coerce_int(usage.get("prompt_tokens"))
        if prompt_tokens is not None:
            tokens["tokens_input"] = prompt_tokens
        completion_tokens = coerce_int(usage.get("completion_tokens"))
        if completion_tokens is not None:
            tokens["tokens_output"] = completion_tokens
    normalized = normalize_usage(info, tokens)
    normalized.setdefault("time_to_first_token", None)
    normalized.setdefault("time_per_output_token", None)
    return normalized


def _extract_stream_usage(chunk: Mapping[str, Any]) -> Mapping[str, Any] | None:
    usage = chunk.get("usage")
    if isinstance(usage, Mapping):
        return usage
    choices = chunk.get("choices")
    if isinstance(choices, Sequence) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            nested_usage = first.get("usage")
            if isinstance(nested_usage, Mapping):
                return nested_usage
    return None


def _extract_choices(obj: Any) -> Sequence[Any] | None:
    choices = obj.get("choices") if isinstance(obj, Mapping) else getattr(obj, "choices", None)
    if isinstance(choices, Sequence):
        return choices
    return None


def _extract_delta(choice: Any) -> Any:
    if isinstance(choice, Mapping):
        return choice.get("delta")
    return getattr(choice, "delta", None)


def _extract_content(delta: Any) -> str:
    if isinstance(delta, Mapping):
        content = delta.get("content")
    else:
        content = getattr(delta, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, Iterable):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)
    return ""


@register_model_client(
    name="client:openai/chat",
    metadata=OPENAI_CHAT_INFO,
)
class OpenAIChatClient(BaseChatClient):
    """Adapter around the OpenAI Python client."""

    info: ClassVar[ModelClientInfo] = OPENAI_CHAT_INFO

    def __init__(
        self,
        *,
        default_model: str = "gpt-3.5-turbo",
        api_key: str | SecretStr | None = None,
        organization: str | None = None,
    ) -> None:
        self._client: Any | None = None
        self._default_model = default_model
        if isinstance(api_key, SecretStr):
            secret = api_key
        elif isinstance(api_key, str):
            secret = SecretStr(api_key)
        else:
            secret = None
        self._api_key: SecretStr | None = secret
        self._organization = organization

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            openai_module = importlib.import_module("openai")
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise ModelClientError("openai package is required to use OpenAIChatClient") from exc

        env_api_key = get_env(OPENAI_API_KEY_ENV)
        if env_api_key and self._api_key is None:
            self._api_key = SecretStr(env_api_key)
        if self._api_key is None:
            raise AuthenticationError(f"Environment variable '{OPENAI_API_KEY_ENV}' is required")
        organization = self._organization or get_env(OPENAI_ORG_ID_ENV)
        try:
            client = openai_module.OpenAI(
                api_key=self._api_key.get_secret_value(),
                organization=organization,
            )
        except Exception as exc:  # pragma: no cover - SDK construction issues
            raise AuthenticationError("Failed to initialize OpenAI client") from exc
        self._client = client
        return client

    def validate_credentials(self) -> None:
        self._ensure_client()

    def invoke(self, *, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        client = self._ensure_client()
        request_kwargs = dict(kwargs)
        model = request_kwargs.pop("model", self._default_model)
        stream = bool(request_kwargs.pop("stream", False))

        if stream:
            chunks = list(
                stream_with_telemetry(
                    self._iter_stream(
                        messages=messages, model=model, request_kwargs=request_kwargs
                    ),
                    info=self.info,
                )
            )
            if chunks:
                content = "".join(
                    chunk.delta.content for chunk in chunks if chunk.delta and chunk.delta.content
                )
                message = Message(role=Role.ASSISTANT, content=content)
                last_chunk = chunks[-1]
                return ChatResponse(message=message, raw=last_chunk.raw, usage=last_chunk.telemetry)
            message = Message(role=Role.ASSISTANT, content="")
            return ChatResponse(message=message, raw=None, usage=None)

        def call_impl() -> ChatResponse:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=_format_messages(messages),
                    stream=False,
                    **request_kwargs,
                )
            except Exception as exc:  # pragma: no cover - transport errors
                raise ModelClientError("OpenAI chat request failed") from exc

            choice = response["choices"][0]
            message = _extract_message(choice)
            usage = _convert_openai_usage(
                self.info, cast(Mapping[str, Any] | None, response.get("usage"))
            )
            return ChatResponse(message=message, raw=response, usage=usage)

        call = OPENAI_RETRY_DECORATOR(call_impl)
        return cast(ChatResponse, invoke_with_telemetry(self.info, call))

    def stream(self, *, messages: Sequence[Message], **kwargs: Any) -> Iterator[ChatChunk]:
        request_kwargs = dict(kwargs)
        model = request_kwargs.pop("model", self._default_model)
        yield from stream_with_telemetry(
            self._iter_stream(messages=messages, model=model, request_kwargs=request_kwargs),
            info=self.info,
        )

    def _iter_stream(
        self, *, messages: Sequence[Message], model: str, request_kwargs: Mapping[str, Any]
    ) -> Iterator[ChatChunk]:
        client = self._ensure_client()
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=_format_messages(messages),
                stream=True,
                **dict(request_kwargs),
            )
        except Exception as exc:  # pragma: no cover - transport errors
            raise ModelClientError(f"OpenAI chat stream failed with error: {exc!s}") from exc

        for chunk in stream:
            choices = _extract_choices(chunk)
            delta = ""
            if choices:
                first_choice = choices[0]
                delta_payload = _extract_delta(first_choice)
                if delta_payload is not None:
                    delta = _extract_content(delta_payload)
            if isinstance(chunk, Mapping):
                raw_chunk: dict[str, Any] = dict(chunk)
            elif hasattr(chunk, "model_dump") and callable(chunk.model_dump):
                raw_chunk = cast(dict[str, Any], chunk.model_dump())
            elif hasattr(chunk, "to_dict") and callable(chunk.to_dict):
                raw_chunk = cast(dict[str, Any], chunk.to_dict())
            else:
                raw_chunk = {"payload": chunk}
            usage = _extract_stream_usage(raw_chunk)
            if usage is not None:
                raw_chunk["usage"] = _convert_openai_usage(self.info, usage)
            yield ChatChunk(delta=Message(role=Role.ASSISTANT, content=delta), raw=raw_chunk)

    def shutdown(self) -> None:
        self._client = None

    def get_init_params(self) -> Mapping[str, Any]:
        """Return initialization parameters for cache key generation."""
        params: dict[str, Any] = {
            "default_model": self._default_model,
        }
        # Include organization if set (affects API routing)
        if self._organization:
            params["organization"] = self._organization
        return params


__all__ = ["OpenAIChatClient"]
