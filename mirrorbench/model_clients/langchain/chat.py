"""LangChain ChatModel wrapper."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from importlib import import_module
from typing import Any, ClassVar, cast

from tenacity import retry, stop_after_attempt, wait_exponential

from mirrorbench.core.constants import OPENAI_API_KEY_ENV
from mirrorbench.core.models.messages import Message, Role
from mirrorbench.core.models.registry import ModelClientInfo
from mirrorbench.core.registry.decorators import register_model_client
from mirrorbench.model_clients.base import BaseChatClient, ChatChunk, ChatResponse
from mirrorbench.model_clients.exceptions import AuthenticationError, ModelClientError
from mirrorbench.model_clients.telemetry import invoke_with_telemetry, stream_with_telemetry

LANGCHAIN_RETRY_WAIT = wait_exponential(multiplier=2, min=1, max=60)
LANGCHAIN_RETRY_MAX_ATTEMPTS = 5
LANGCHAIN_RETRY_DECORATOR = cast(
    Callable[[Callable[[], ChatResponse]], Callable[[], ChatResponse]],
    retry(
        wait=LANGCHAIN_RETRY_WAIT,
        stop=stop_after_attempt(LANGCHAIN_RETRY_MAX_ATTEMPTS),
        reraise=True,
    ),
)

LANGCHAIN_CHAT_INFO = ModelClientInfo(
    name="client:langchain/chat",
    provider="langchain",
    capabilities={"chat"},
)


@register_model_client(
    name="client:langchain/chat",
    metadata=LANGCHAIN_CHAT_INFO,
)
class LangChainChatClient(BaseChatClient):
    """Adapter around LangChain chat models."""

    info: ClassVar[ModelClientInfo] = LANGCHAIN_CHAT_INFO

    def __init__(
        self,
        chat_model: Any | None = None,
        *,
        model_import: str | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if chat_model is not None and (model_import is not None or model_kwargs is not None):
            raise ModelClientError(
                "Provide either 'chat_model' or model import parameters, not both"
            )
        if chat_model is None:
            if not model_import:
                raise ModelClientError(
                    "LangChainChatClient requires 'chat_model' or 'model_import'"
                )
            chat_model = self._initialize_model(model_import, model_kwargs)
        if not hasattr(chat_model, "invoke"):
            raise ModelClientError("LangChain chat model must provide an 'invoke' method")
        self._chat_model = chat_model
        # Store initialization params for cache key generation
        self._model_import = model_import
        self._model_kwargs = dict(model_kwargs) if model_kwargs else None

    def invoke(self, *, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        def call_impl() -> ChatResponse:
            try:
                converted = self._to_langchain_messages(messages)
                result = self._chat_model.invoke(converted, **kwargs)
            except Exception as exc:  # pragma: no cover - delegate errors
                raise ModelClientError(
                    f"LangChain chat invocation failed with error: {exc!s}"
                ) from exc
            content = getattr(result, "content", str(result))
            usage = self._extract_usage(result)
            return ChatResponse(message=Message(role=Role.ASSISTANT, content=content), usage=usage)

        call = LANGCHAIN_RETRY_DECORATOR(call_impl)
        return cast(ChatResponse, invoke_with_telemetry(self.info, call))

    def stream(self, *, messages: Sequence[Message], **kwargs: Any) -> Iterator[ChatChunk]:
        if not hasattr(self._chat_model, "stream"):
            yield from super().stream(messages=messages, **kwargs)
            return

        def iterator() -> Iterator[ChatChunk]:
            try:
                converted = self._to_langchain_messages(messages)
                for chunk in self._chat_model.stream(converted, **kwargs):
                    content = getattr(chunk, "content", str(chunk))
                    yield ChatChunk(delta=Message(role=Role.ASSISTANT, content=content))
            except Exception as exc:  # pragma: no cover - delegate errors
                raise ModelClientError("LangChain chat streaming failed") from exc

        yield from stream_with_telemetry(iterator(), info=self.info)

    def validate_credentials(self) -> None:
        module_path = getattr(self._chat_model.__class__, "__module__", "")
        if "ChatOpenAI" in module_path and not os.getenv(OPENAI_API_KEY_ENV):
            raise AuthenticationError(
                f"Environment variable '{OPENAI_API_KEY_ENV}' is required for OpenAI models"
            )
        validator = getattr(self._chat_model, "validate_credentials", None)
        if callable(validator):
            validator()

    def get_init_params(self) -> Mapping[str, Any]:
        """Return initialization parameters for cache key generation."""
        params: dict[str, Any] = {}

        # Include model import path if available
        if self._model_import:
            params["model_import"] = self._model_import

        # Include model kwargs if available (these contain critical config like model name)
        if self._model_kwargs:
            params["model_kwargs"] = self._model_kwargs

        # If chat_model was passed directly, try to extract identifying info
        if not self._model_import:
            # Try to get model name or class info
            model_class = self._chat_model.__class__
            params["model_class"] = f"{model_class.__module__}.{model_class.__name__}"

            # Try to extract model-specific config
            if hasattr(self._chat_model, "model_name"):
                params["model_name"] = self._chat_model.model_name
            elif hasattr(self._chat_model, "model"):
                params["model"] = self._chat_model.model

            # Extract other relevant attributes that affect behavior
            for attr in ["temperature", "max_tokens", "top_p"]:
                if hasattr(self._chat_model, attr):
                    val = getattr(self._chat_model, attr)
                    if val is not None:
                        params[attr] = val

        return params

    @staticmethod
    def _initialize_model(model_import: str, model_kwargs: Mapping[str, Any] | None) -> Any:
        if "." not in model_import:
            raise ModelClientError("model_import must be of the form 'module.Class'")
        module_path, _, attr_name = model_import.rpartition(".")
        try:
            module = import_module(module_path)
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise ModelClientError(f"Failed to import module '{module_path}'") from exc
        try:
            constructor = getattr(module, attr_name)
        except AttributeError as exc:  # pragma: no cover - defensive fallback
            raise ModelClientError(
                f"Module '{module_path}' has no attribute '{attr_name}'"
            ) from exc
        if not callable(constructor):
            raise ModelClientError(f"Target '{model_import}' is not callable")
        if model_kwargs is None:
            kwargs: Mapping[str, Any] = {}
        elif isinstance(model_kwargs, Mapping):
            kwargs = model_kwargs
        else:
            raise ModelClientError("model_kwargs must be a mapping if provided")
        try:
            instance = constructor(**dict(kwargs))
        except Exception as exc:  # pragma: no cover - delegate instantiation errors
            raise ModelClientError(f"Failed to instantiate '{model_import}'") from exc
        return instance

    @staticmethod
    def _extract_usage(result: Any) -> Mapping[str, Any] | None:
        usage = getattr(result, "usage_metadata", None)
        if isinstance(usage, Mapping):
            normalized: dict[str, Any] = {}
            if "input_tokens" in usage:
                normalized["tokens_input"] = usage.get("input_tokens")
            if "output_tokens" in usage:
                normalized["tokens_output"] = usage.get("output_tokens")
            if normalized:
                return normalized
        metadata = getattr(result, "response_metadata", None)
        if isinstance(metadata, Mapping):
            token_usage = metadata.get("token_usage")
            if isinstance(token_usage, Mapping):
                normalized = {
                    "tokens_input": token_usage.get("input_tokens"),
                    "tokens_output": token_usage.get("output_tokens"),
                }
                if any(value is not None for value in normalized.values()):
                    return normalized
        return None

    @staticmethod
    def _to_langchain_messages(messages: Sequence[Message]) -> list[Any]:
        (
            ai_message_cls,
            human_message_cls,
            system_message_cls,
            tool_message_cls,
        ) = _ensure_langchain_messages()

        converted: list[Any] = []
        for message in messages:
            name = message.name
            metadata = message.metadata or {}
            metadata_mapping = metadata if isinstance(metadata, Mapping) else {}
            if message.role is Role.USER:
                converted.append(human_message_cls(content=message.content, name=name))
            elif message.role is Role.ASSISTANT:
                converted.append(
                    ai_message_cls(
                        content=message.content, name=name, additional_kwargs=dict(metadata_mapping)
                    )
                )
            elif message.role is Role.SYSTEM:
                converted.append(system_message_cls(content=message.content, name=name))
            elif message.role is Role.TOOL:
                tool_call_id = metadata_mapping.get("tool_call_id")
                converted.append(
                    tool_message_cls(
                        content=message.content, tool_call_id=tool_call_id or "", name=name
                    )
                )
            else:
                converted.append(system_message_cls(content=message.content, name=name))
        return converted


__all__ = ["LangChainChatClient"]


def _ensure_langchain_messages() -> tuple[Any, Any, Any, Any]:
    try:
        messages_module = import_module("langchain_core.messages")
    except ImportError as exc:  # pragma: no cover - optional dependency missing
        raise ModelClientError(
            "langchain-core package is required for LangChainChatClient"
        ) from exc

    return (
        messages_module.AIMessage,
        messages_module.HumanMessage,
        messages_module.SystemMessage,
        messages_module.ToolMessage,
    )
