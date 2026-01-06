"""Caching wrapper for chat model clients."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from mirrorbench.cache.keys import build_model_client_cache_key
from mirrorbench.cache.manager import CacheManager
from mirrorbench.core.models.cache import CacheKey
from mirrorbench.core.models.messages import Message, Role
from mirrorbench.model_clients.base import ChatClient, ChatResponse


class CachingChatClient(ChatClient):
    """Wraps a chat client with cache lookups for ``invoke`` calls."""

    def __init__(
        self,
        *,
        delegate: ChatClient,
        manager: CacheManager,
        namespace: str,
        ttl_seconds: int | None,
    ) -> None:
        self._delegate = delegate
        self._manager = manager
        self._namespace = namespace
        self._ttl_seconds = ttl_seconds
        self.info = delegate.info

    # ------------------------------------------------------------------
    # ChatClient interface
    # ------------------------------------------------------------------
    def invoke(self, *, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        control: dict[str, Any] = {}
        for param_name in list(kwargs.keys()):
            if param_name.startswith("_cache_"):
                control[param_name] = kwargs.pop(param_name)

        skip_cache = bool(control.get("_cache_disable", False)) or not self._manager.enabled
        ttl_override_raw = control.get("_cache_ttl")
        ttl_override: int | None
        if ttl_override_raw is not None:
            try:
                ttl_override = int(ttl_override_raw)
            except (TypeError, ValueError):
                ttl_override = None
        else:
            ttl_override = None

        request_params: Mapping[str, Any] = dict(kwargs)

        if skip_cache:
            response = self._delegate.invoke(messages=messages, **kwargs)
            _annotate_usage(response, cache_hit=False)
            return response

        # Get initialization params from delegate client for cache key
        init_params: Mapping[str, Any] | None = None
        if hasattr(self._delegate, "get_init_params"):
            try:
                init_params = self._delegate.get_init_params()
            except Exception:  # pragma: no cover - defensive fallback
                # If get_init_params fails, proceed with None (backward compat)
                init_params = None

        key: CacheKey = build_model_client_cache_key(
            messages=messages,
            request_params=request_params,
            model_info=self.info,
            namespace=self._namespace,
            init_params=init_params,
        )
        cached = self._manager.get(key)
        if cached is not None:
            response = _response_from_bytes(cached.value)
            _annotate_usage(response, cache_hit=True)
            return response

        response = self._delegate.invoke(messages=messages, **kwargs)
        _annotate_usage(response, cache_hit=False)
        payload = _response_to_bytes(response)
        self._manager.set(
            key,
            payload,
            metadata={"model": self.info.name, "provider": self.info.provider},
            ttl_seconds=ttl_override if ttl_override is not None else self._ttl_seconds,
        )
        return response

    async def invoke_async(self, *, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        control: dict[str, Any] = {}
        for param_name in list(kwargs.keys()):
            if param_name.startswith("_cache_"):
                control[param_name] = kwargs.pop(param_name)

        skip_cache = bool(control.get("_cache_disable", False)) or not self._manager.enabled
        ttl_override_raw = control.get("_cache_ttl")
        ttl_override: int | None
        if ttl_override_raw is not None:
            try:
                ttl_override = int(ttl_override_raw)
            except (TypeError, ValueError):
                ttl_override = None
        else:
            ttl_override = None

        request_params: Mapping[str, Any] = dict(kwargs)

        if skip_cache:
            response = await self._invoke_delegate_async(messages=messages, **kwargs)
            _annotate_usage(response, cache_hit=False)
            return response

        init_params: Mapping[str, Any] | None = None
        if hasattr(self._delegate, "get_init_params"):
            try:
                init_params = self._delegate.get_init_params()
            except Exception:  # pragma: no cover - defensive fallback
                init_params = None

        key: CacheKey = build_model_client_cache_key(
            messages=messages,
            request_params=request_params,
            model_info=self.info,
            namespace=self._namespace,
            init_params=init_params,
        )
        cached = self._manager.get(key)
        if cached is not None:
            response = _response_from_bytes(cached.value)
            _annotate_usage(response, cache_hit=True)
            return response

        response = await self._invoke_delegate_async(messages=messages, **kwargs)
        _annotate_usage(response, cache_hit=False)
        payload = _response_to_bytes(response)
        self._manager.set(
            key,
            payload,
            metadata={"model": self.info.name, "provider": self.info.provider},
            ttl_seconds=ttl_override if ttl_override is not None else self._ttl_seconds,
        )
        return response

    def stream(self, *, messages: Sequence[Message], **kwargs: Any) -> Iterator[Any]:
        # Streaming is forwarded directly without caching.
        yield from self._delegate.stream(messages=messages, **kwargs)

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------
    def get_init_params(self) -> Mapping[str, Any]:
        """Delegate to underlying client's get_init_params."""
        if hasattr(self._delegate, "get_init_params"):
            return self._delegate.get_init_params()
        return {}

    async def _invoke_delegate_async(
        self, *, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponse | Any:
        async_invoke = getattr(self._delegate, "invoke_async", None)
        if callable(async_invoke):
            response = async_invoke(messages=messages, **kwargs)
            if inspect.isawaitable(response):
                return await response
            return response
        return await asyncio.to_thread(self._delegate.invoke, messages=messages, **kwargs)

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - delegation helper
        return getattr(self._delegate, name)


# ----------------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------------


def _response_to_bytes(response: ChatResponse) -> bytes:
    payload = {
        "message": _message_to_dict(response.message),
        "usage": _safe_mapping(response.usage),
        "raw": _safe_mapping(response.raw),
    }
    return json.dumps(payload, sort_keys=True, default=str).encode("utf-8")


def _response_from_bytes(payload: bytes) -> ChatResponse:
    data = json.loads(payload.decode("utf-8"))
    message_dict = data.get("message", {})
    message = Message(
        role=Role(message_dict.get("role", "assistant")),
        content=message_dict.get("content", ""),
        name=message_dict.get("name"),
        metadata=message_dict.get("metadata", {}),
    )
    usage = data.get("usage")
    raw = data.get("raw")
    return ChatResponse(message=message, usage=usage, raw=raw)


def _message_to_dict(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role.value,
        "content": message.content,
    }
    if message.name:
        payload["name"] = message.name
    if message.metadata:
        payload["metadata"] = message.metadata
    return payload


def _safe_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        try:
            json.dumps(value)
        except TypeError:
            return {k: str(v) for k, v in value.items()}
        return dict(value)
    return None


def _annotate_usage(response: ChatResponse, *, cache_hit: bool) -> None:
    usage = dict(response.usage or {})
    usage["cache_hit"] = cache_hit
    response.usage = usage


__all__ = ["CachingChatClient"]
