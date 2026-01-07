"""Provider model clients shipped with MirrorBench."""

from __future__ import annotations

import importlib

from mirrorbench.core.registry import registry
from mirrorbench.core.registry.components import BaseModelClient
from mirrorbench.core.registry.decorators import register_model_client

# Trigger registration of built-in clients
from mirrorbench.model_clients.base import BaseChatClient, ChatChunk, ChatClient, ChatResponse
from mirrorbench.model_clients.caching_wrapper import CachingChatClient

importlib.import_module("mirrorbench.model_clients.registry")

__all__ = [
    "BaseChatClient",
    "BaseModelClient",
    "CachingChatClient",
    "ChatChunk",
    "ChatClient",
    "ChatResponse",
    "register_model_client",
    "registry",
]
