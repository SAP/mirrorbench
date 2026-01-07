"""User proxy adapters packaged with MirrorBench."""

from __future__ import annotations

from mirrorbench.adapters import registry as _registry  # noqa: F401
from mirrorbench.adapters.llm import LLMAdapter
from mirrorbench.core.models.registry import UserProxyAdapterInfo
from mirrorbench.core.registry import BaseUserProxyAdapter, register_user_proxy

__all__ = [
    "BaseUserProxyAdapter",
    "LLMAdapter",
    "UserProxyAdapterInfo",
    "register_user_proxy",
]
