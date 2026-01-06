"""Helpers for building deterministic cache keys."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

from mirrorbench.core.models.cache import CacheKey
from mirrorbench.core.models.messages import Message
from mirrorbench.core.models.registry import ModelClientInfo


def build_model_client_cache_key(  # noqa: PLR0913
    *,
    messages: Sequence[Message],
    request_params: Mapping[str, Any] | None,
    model_info: ModelClientInfo,
    namespace: str,
    version: str = "1",
    init_params: Mapping[str, Any] | None = None,
) -> CacheKey:
    """Return a deterministic cache key for a chat model invocation.

    Args:
        messages: Sequence of messages for the chat invocation
        request_params: Parameters passed at request time (invoke kwargs)
        model_info: Model client metadata
        namespace: Cache namespace
        version: Cache key version
        init_params: Initialization parameters that affect model behavior
                     (e.g., default_model, model_import, model_kwargs)
    """

    payload = {
        "messages": [_message_payload(msg) for msg in messages],
        "request_params": _canonicalise(request_params or {}),
        "init_params": _canonicalise(init_params or {}),
        "model": model_info.name,
        "provider": model_info.provider,
        "capabilities": sorted(model_info.capabilities) if model_info.capabilities else [],
    }
    normalised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = sha256(normalised.encode("utf-8")).hexdigest()
    metadata = {"model": model_info.name, "provider": model_info.provider}
    return CacheKey(namespace=namespace, key=digest[:32], version=version, metadata=metadata)


def _message_payload(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role.value,
        "content": message.content,
    }
    if message.name:
        payload["name"] = message.name
    if message.metadata:
        payload["metadata"] = _canonicalise(message.metadata)
    return payload


def _canonicalise(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): _canonicalise(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list | tuple):
        return [_canonicalise(v) for v in value]
    if isinstance(value, set | frozenset):
        return sorted(_canonicalise(v) for v in value)
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return str(value)


__all__ = ["build_model_client_cache_key"]
