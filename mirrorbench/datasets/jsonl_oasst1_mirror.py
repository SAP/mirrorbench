"""Dataset implementation for preprocessed OASST1 mirror conversations."""

from __future__ import annotations

from typing import ClassVar

from mirrorbench.core.models.registry import DatasetInfo
from mirrorbench.core.registry import register_dataset
from mirrorbench.datasets.jsonl_mirror_base import MIRROR_TASK_DRIVER_NAME, JSONLMirrorBase

DATASET_NAME = "dataset:jsonl/oasst1_mirror"

OASST1_MIRROR_INFO = DatasetInfo(
    name=DATASET_NAME,
    supported_tasks={"chat", "qa"},
    splits={"default"},
    languages={"en"},
    reference_stats_mode="optional",
    citation="OpenAssistant Conversations Dataset (OASST1) (preprocessed for MirrorBench)",
).model_copy(
    update={
        "task_driver": MIRROR_TASK_DRIVER_NAME,
    }
)


@register_dataset(name=DATASET_NAME, metadata=OASST1_MIRROR_INFO)
class JSONLOASST1Mirror(JSONLMirrorBase):
    """Dataset that reads preprocessed OASST1 mirror conversations from JSONL files.

    This dataset loader expects multi-turn conversations from the OpenAssistant dataset
    that have been preprocessed to include task descriptions and normalized turn
    structures suitable for evaluating user proxies.
    """

    info: ClassVar[DatasetInfo] = OASST1_MIRROR_INFO
    default_dataset_name = "oasst1"

    def _default_task_tag(self) -> str:
        return "chat"


__all__ = ["JSONLOASST1Mirror"]
