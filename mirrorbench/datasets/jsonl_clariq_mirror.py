"""Dataset implementation for preprocessed ClariQ mirror conversations."""

from __future__ import annotations

from typing import ClassVar

from mirrorbench.core.models.registry import DatasetInfo
from mirrorbench.core.registry import register_dataset
from mirrorbench.datasets.jsonl_mirror_base import MIRROR_TASK_DRIVER_NAME, JSONLMirrorBase

DATASET_NAME = "dataset:jsonl/clariq_mirror"

CLARIQ_MIRROR_INFO = DatasetInfo(
    name=DATASET_NAME,
    supported_tasks={"clarification", "info_seeking"},
    splits={"default"},
    languages={"en"},
    reference_stats_mode="optional",
    citation="ClariQ dataset (preprocessed for MirrorBench)",
).model_copy(
    update={
        "task_driver": MIRROR_TASK_DRIVER_NAME,
    }
)


@register_dataset(name=DATASET_NAME, metadata=CLARIQ_MIRROR_INFO)
class JSONLClariQMirror(JSONLMirrorBase):
    """Dataset that reads preprocessed ClariQ mirror conversations from JSONL files.

    This dataset loader expects multi-turn clarification conversations that have been
    preprocessed to include task descriptions and normalized turn structures suitable
    for evaluating user proxies in information-seeking scenarios.
    """

    info: ClassVar[DatasetInfo] = CLARIQ_MIRROR_INFO
    default_dataset_name = "clariq"

    def _default_task_tag(self) -> str:
        return "clarification"


__all__ = ["JSONLClariQMirror"]
