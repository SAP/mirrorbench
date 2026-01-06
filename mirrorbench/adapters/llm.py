"""Generic user proxy adapter backed by registry-provided chat model clients."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from mirrorbench.cache import get_cache_manager
from mirrorbench.core.config import CacheConfig
from mirrorbench.core.constants import DEFAULT_CACHE_TTL_SECONDS, REGISTRY_GROUP_MODEL_CLIENTS
from mirrorbench.core.models.episodes import EpisodeSpec
from mirrorbench.core.models.messages import Message, Role, TurnTelemetry
from mirrorbench.core.models.plan import UserProxySpec
from mirrorbench.core.models.registry import UserProxyAdapterInfo
from mirrorbench.core.registry import BaseUserProxyAdapter, registry
from mirrorbench.core.registry.decorators import register_user_proxy
from mirrorbench.io.paths import Paths
from mirrorbench.model_clients import ChatChunk, ChatClient
from mirrorbench.model_clients.caching_wrapper import CachingChatClient
from mirrorbench.model_clients.telemetry import usage_to_turn_telemetry
from mirrorbench.model_clients.utils import is_chat_client

ADAPTER_NAME = "adapter:generic/llm"

LLM_ADAPTER_INFO = UserProxyAdapterInfo(
    name=ADAPTER_NAME,
    capabilities={"chat"},
    description="Generic adapter that delegates to registry-provided chat model clients.",
)


@dataclass(slots=True)
class GenerationResult:
    """Normalized result returned by :class:`LLMSession.generate`."""

    message: Message
    raw: Mapping[str, Any] | None
    usage: Mapping[str, Any] | None
    telemetry: TurnTelemetry


class LLMSession:
    """Runtime session that mediates between the execution engine and a model client."""

    def __init__(
        self,
        *,
        model_client: ChatClient,
        default_kwargs: Mapping[str, Any] | None = None,
        system_prompt: str | None = None,
        combine_system_and_history: bool = False,
        proxy_spec: UserProxySpec | None = None,
    ) -> None:
        self._client = model_client
        self._default_kwargs = dict(default_kwargs or {})
        self._system_message: Message | None = None
        self._combine_system_and_history = combine_system_and_history
        self._proxy_spec = proxy_spec
        if system_prompt:
            self._system_message = Message(role=Role.SYSTEM, content=system_prompt)

    @property
    def client(self) -> ChatClient:
        """Return the underlying chat client."""

        return self._client

    def generate(
        self,
        *,
        turn: EpisodeSpec,
        request_params: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        """Invoke the underlying client and return the normalized result."""

        return self.invoke(turn=turn, request_params=request_params)

    async def generate_async(
        self,
        *,
        turn: EpisodeSpec,
        request_params: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        """Invoke the underlying client asynchronously when available."""

        return await self.invoke_async(turn=turn, request_params=request_params)

    def invoke(
        self,
        *,
        turn: EpisodeSpec,
        request_params: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        conversation = self._build_messages(turn=turn)
        kwargs = dict(self._default_kwargs)
        if request_params:
            kwargs.update(request_params)
        chat_response = self._client.invoke(messages=conversation, **kwargs)
        return self._normalize_response(chat_response)

    async def invoke_async(
        self,
        *,
        turn: EpisodeSpec,
        request_params: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        conversation = self._build_messages(turn=turn)
        kwargs = dict(self._default_kwargs)
        if request_params:
            kwargs.update(request_params)

        async_invoke = getattr(self._client, "invoke_async", None)
        if callable(async_invoke):
            response = async_invoke(messages=conversation, **kwargs)
            if inspect.isawaitable(response):
                chat_response = await response
            else:
                chat_response = response
        else:
            chat_response = await asyncio.to_thread(
                self._client.invoke, messages=conversation, **kwargs
            )
        return self._normalize_response(chat_response)

    def _normalize_response(self, chat_response: Any) -> GenerationResult:
        telemetry = usage_to_turn_telemetry(
            chat_response.usage, provider=self._client.info.provider
        ) or TurnTelemetry(provider=self._client.info.provider)
        if telemetry.metadata is None:
            telemetry.metadata = {}
        telemetry.metadata["component"] = "user_proxy"

        if self._proxy_spec is not None:
            telemetry.metadata["proxy_name"] = self._proxy_spec.name
            telemetry.metadata["proxy_adapter"] = self._proxy_spec.adapter
            if self._proxy_spec.params:
                params = self._proxy_spec.params
                if "model_client" in params:
                    telemetry.metadata["model_client"] = params["model_client"]
                if "client_params" in params and isinstance(params["client_params"], Mapping):
                    client_params = params["client_params"]
                    if "model" in client_params:
                        telemetry.metadata["model"] = client_params["model"]
                    if "model_name" in client_params:
                        telemetry.metadata["model_name"] = client_params["model_name"]
                    if "azure_deployment" in client_params:
                        telemetry.metadata["azure_deployment"] = client_params["azure_deployment"]
                if "request_params" in params and isinstance(params["request_params"], Mapping):
                    request_params = params["request_params"]
                    if "temperature" in request_params:
                        telemetry.metadata["temperature"] = request_params["temperature"]
                    if "max_tokens" in request_params:
                        telemetry.metadata["max_tokens"] = request_params["max_tokens"]

        return GenerationResult(
            message=chat_response.message,
            raw=chat_response.raw,
            usage=chat_response.usage,
            telemetry=telemetry,
        )

    def stream(
        self,
        *,
        turn: EpisodeSpec,
        request_params: Mapping[str, Any] | None = None,
    ) -> Iterator[ChatChunk]:
        """Stream incremental chunks from the underlying client."""

        conversation = self._build_messages(turn=turn)
        kwargs = dict(self._default_kwargs)
        if request_params:
            kwargs.update(request_params)
        yield from self._client.stream(messages=conversation, **kwargs)

    def shutdown(self) -> None:
        """Release resources held by the underlying client, if any."""

        shutdown = getattr(self._client, "shutdown", None)
        if callable(shutdown):
            shutdown()

    def validate_credentials(self) -> None:
        """Forward credential validation to the underlying client when available."""

        validator = getattr(self._client, "validate_credentials", None)
        if callable(validator):
            validator()

    def _build_messages(
        self,
        *,
        turn: EpisodeSpec,
    ) -> list[Message]:
        conversation: list[Message] = []

        # If combine_system_and_history is enabled, merge everything into a single user message
        if self._combine_system_and_history:
            # Build the combined content
            combined_parts = []

            # Add system prompt if available
            if self._system_message is not None:
                combined_parts.append(self._system_message.content)

            # Add all messages from chat history
            for msg in turn.chat_history:
                if msg.role == Role.SYSTEM:
                    # Include system messages from history
                    combined_parts.append(msg.content)
                elif msg.role == Role.USER:
                    combined_parts.append(f"User: {msg.content}")
                elif msg.role == Role.ASSISTANT:
                    combined_parts.append(f"Assistant: {msg.content}")

            # Create a single user message with all the content
            combined_content = "\n\n".join(combined_parts)
            conversation.append(Message(role=Role.USER, content=combined_content))
        else:
            # Original behavior: system message first, then history
            if self._system_message is not None:
                conversation.append(self._system_message)
            conversation.extend(list(turn.chat_history))

        return conversation


@register_user_proxy(name=ADAPTER_NAME, metadata=LLM_ADAPTER_INFO)
class LLMAdapter(BaseUserProxyAdapter):
    """Adapter that materialises sessions backed by chat-oriented model clients."""

    info: ClassVar[UserProxyAdapterInfo] = LLM_ADAPTER_INFO

    def __init__(self) -> None:
        self._cache_config = CacheConfig(
            enabled=True,
            ttl_seconds=DEFAULT_CACHE_TTL_SECONDS,
            backend="sqlite",
        )
        self._paths: Paths | None = None

    def configure_cache(self, *, cache_config: CacheConfig, paths: Paths) -> None:
        self._cache_config = cache_config
        self._paths = paths

    def spawn(self, *, config: UserProxySpec, run_id: str) -> LLMSession:
        _ = run_id  # Explicitly acknowledge the run identifier is unused for now.
        params = dict(config.params or {})
        model_client_name = params.get("model_client")
        if not isinstance(model_client_name, str) or not model_client_name:
            msg = "LLMAdapter requires 'model_client' (str) in config.params"
            raise ValueError(msg)

        client_params = params.get("client_params", {})
        if client_params is None:
            client_params = {}
        if not isinstance(client_params, Mapping):
            msg = "'client_params' must be a mapping when provided"
            raise TypeError(msg)

        request_params = params.get("request_params", {})
        if request_params is None:
            request_params = {}
        if not isinstance(request_params, Mapping):
            msg = "'request_params' must be a mapping when provided"
            raise TypeError(msg)

        system_prompt = params.get("system_prompt")
        if system_prompt is not None and not isinstance(system_prompt, str):
            msg = "'system_prompt' must be a string when provided"
            raise TypeError(msg)

        combine_system_and_history = params.get("combine_system_and_history", False)
        if not isinstance(combine_system_and_history, bool):
            msg = "'combine_system_and_history' must be a boolean when provided"
            raise TypeError(msg)

        factory = registry.factory(REGISTRY_GROUP_MODEL_CLIENTS, model_client_name)
        model_client = factory(**dict(client_params))
        if not is_chat_client(model_client):
            msg = "Model client must expose 'invoke' and 'stream' callables"
            raise TypeError(msg)

        if self._cache_config.enabled:
            paths = self._paths or Paths.default()
            manager = get_cache_manager(paths, self._cache_config)
            if manager.enabled:
                client_namespace = (
                    f"model_client:{getattr(model_client.info, 'name', model_client_name)}"
                )
                model_client = CachingChatClient(
                    delegate=model_client,
                    manager=manager,
                    namespace=client_namespace,
                    ttl_seconds=self._cache_config.ttl_seconds,
                )

        return LLMSession(
            model_client=model_client,
            default_kwargs=dict(request_params),
            system_prompt=system_prompt,
            combine_system_and_history=combine_system_and_history,
            proxy_spec=config,
        )


# ruff: noqa: RUF022
__all__ = [
    "GenerationResult",
    "LLMAdapter",
    "LLM_ADAPTER_INFO",
    "LLMSession",
]
