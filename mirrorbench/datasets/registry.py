"""Register built-in datasets with the global registry."""

from __future__ import annotations

from mirrorbench.datasets.jsonl_chatbot_arena_mirror import JSONLChatbotArenaMirror
from mirrorbench.datasets.jsonl_clariq_mirror import JSONLClariQMirror
from mirrorbench.datasets.jsonl_oasst1_mirror import JSONLOASST1Mirror
from mirrorbench.datasets.jsonl_qulac_mirror import JSONLQulacMirror

# ruff: noqa: RUF022
__all__ = [
    "JSONLChatbotArenaMirror",
    "JSONLClariQMirror",
    "JSONLOASST1Mirror",
    "JSONLQulacMirror",
]
