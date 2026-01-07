"""Dataset implementation for preprocessed Qulac mirror conversations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from mirrorbench.core.models.episodes import EpisodeSpec
from mirrorbench.core.models.plan import DatasetSpec
from mirrorbench.core.models.registry import DatasetInfo
from mirrorbench.core.registry import register_dataset
from mirrorbench.datasets.jsonl_mirror_base import MIRROR_TASK_DRIVER_NAME, JSONLMirrorBase

DATASET_NAME = "dataset:jsonl/qulac_mirror"

QULAC_MIRROR_INFO = DatasetInfo(
    name=DATASET_NAME,
    supported_tasks={"clarification", "query_clarification"},
    splits={"default"},
    languages={"en"},
    reference_stats_mode="optional",
    citation="Qulac dataset (preprocessed for MirrorBench)",
).model_copy(
    update={
        "task_driver": MIRROR_TASK_DRIVER_NAME,
    }
)


@register_dataset(name=DATASET_NAME, metadata=QULAC_MIRROR_INFO)
class JSONLQulacMirror(JSONLMirrorBase):
    """Dataset that reads preprocessed Qulac mirror conversations from JSONL files.

    This dataset loader expects query clarification conversations that have been
    preprocessed to include task descriptions and normalized turn structures suitable
    for evaluating user proxies in query clarification scenarios.
    """

    info: ClassVar[DatasetInfo] = QULAC_MIRROR_INFO
    default_dataset_name = "qulac"

    def _default_task_tag(self) -> str:
        return "query_clarification"

    def build_episode(
        self,
        *,
        record: Mapping[str, Any],
        spec: DatasetSpec,
        split: str,
        index: int,
    ) -> EpisodeSpec:
        episode = super().build_episode(record=record, spec=spec, split=split, index=index)
        return EpisodeSpec(
            episode_id=episode.episode_id,
            task_tag=episode.task_tag,
            chat_history=episode.chat_history,
            references={},
            metadata=episode.metadata,
            reference_stats=episode.reference_stats,
        )


__all__ = ["JSONLQulacMirror"]
