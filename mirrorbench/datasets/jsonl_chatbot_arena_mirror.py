"""Dataset implementation for preprocessed Chatbot Arena mirror conversations."""

from __future__ import annotations

from typing import ClassVar

from mirrorbench.core.models.registry import DatasetInfo
from mirrorbench.core.registry import register_dataset
from mirrorbench.datasets.jsonl_mirror_base import MIRROR_TASK_DRIVER_NAME, JSONLMirrorBase

DATASET_NAME = "dataset:jsonl/chatbot_arena_mirror"

CHATBOT_ARENA_MIRROR_INFO = DatasetInfo(
    name=DATASET_NAME,
    supported_tasks={"chat_arena"},
    splits={"default"},
    languages={"en"},
    reference_stats_mode="optional",
    citation="LMSYS Chatbot Arena conversations (preprocessed for MirrorBench)",
).model_copy(
    update={
        "task_driver": MIRROR_TASK_DRIVER_NAME,
    }
)


@register_dataset(name=DATASET_NAME, metadata=CHATBOT_ARENA_MIRROR_INFO)
class JSONLChatbotArenaMirror(JSONLMirrorBase):
    """Dataset that reads preprocessed Chatbot Arena mirror conversations from JSONL files.

    This dataset loader expects conversations that have been preprocessed to include
    task descriptions and normalized turn structures suitable for evaluating user proxies.
    """

    info: ClassVar[DatasetInfo] = CHATBOT_ARENA_MIRROR_INFO
    default_dataset_name = "chatbot_arena"


__all__ = ["JSONLChatbotArenaMirror"]
