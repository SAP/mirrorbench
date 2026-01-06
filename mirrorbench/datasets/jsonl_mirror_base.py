"""Base class for JSONL mirror dataset implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mirrorbench.core.models.episodes import EpisodeSpec
from mirrorbench.core.models.messages import Message, Role
from mirrorbench.core.models.plan import DatasetSpec
from mirrorbench.datasets.base_dataset import BaseDataset, DatasetError

MIRROR_TASK_DRIVER_NAME = "task:mirror/conversation"


class JSONLMirrorBase(BaseDataset):
    """Base class for mirror conversation datasets stored in JSONL format.

    This class provides common functionality for parsing preprocessed mirror datasets
    that contain task descriptions and normalized turn structures.

    Subclasses only need to:
    1. Define dataset metadata (name, tasks, languages, etc.)
    2. Override default_dataset_name if the dataset field differs
    """

    loader_name = "jsonl"
    pipeline_version = "1"
    default_dataset_name: str = ""  # Subclasses should override this

    def build_loader_params(self, spec: DatasetSpec) -> Mapping[str, Any]:
        loader_params: dict[str, Any] = dict(super().build_loader_params(spec))
        if "path" not in loader_params and "splits" not in loader_params:
            path = spec.params.get("path")
            if not path:
                msg = (
                    f"{self.info.name} dataset requires either loader.path or dataset path "
                    "parameter"
                )
                raise DatasetError(msg)
            loader_params["path"] = path
        for key in ("encoding",):
            if key in spec.params and key not in loader_params:
                loader_params[key] = spec.params[key]
        return loader_params

    def build_episode(
        self,
        *,
        record: Mapping[str, Any],
        spec: DatasetSpec,
        split: str,
        index: int,
    ) -> EpisodeSpec:
        episode_id = self._extract_episode_id(record, split, index)
        task_tag = str(record.get("task_tag", spec.task or self._default_task_tag()))

        # Convert turns to chat history (reference conversation)
        chat_history = self._parse_turns(record, episode_id)

        # Build metadata
        metadata = self._build_metadata(record, split)

        references = {
            "real_conversation": [
                {
                    "role": message.role.value,
                    "content": message.content,
                    "metadata": dict(message.metadata),
                }
                for message in chat_history
            ]
        }

        return EpisodeSpec(
            episode_id=episode_id,
            task_tag=task_tag,
            chat_history=chat_history,
            references=references,
            metadata=metadata,
        )

    def _extract_episode_id(self, record: Mapping[str, Any], split: str, index: int) -> str:
        """Extract episode ID from record. Subclasses can override for custom logic."""
        dataset_name = self.default_dataset_name
        if not dataset_name:
            raise DatasetError("Subclasses must define default_dataset_name")
        return str(
            record.get("conversation_id") or record.get("id") or f"{dataset_name}-{split}-{index}"
        )

    def _default_task_tag(self) -> str:
        """Return the default task tag. Subclasses can override."""
        # Return the first supported task as default
        if self.info.supported_tasks:
            return next(iter(self.info.supported_tasks))
        return "unknown"

    def _parse_turns(self, record: Mapping[str, Any], episode_id: str) -> list[Message]:
        """Parse turns from record and convert to Message objects."""
        turns = record.get("turns", [])
        if not turns:
            msg = f"No turns found in episode {episode_id}"
            raise DatasetError(msg)

        chat_history: list[Message] = []
        for turn_idx, turn in enumerate(turns):
            if not isinstance(turn, Mapping):
                raise DatasetError(f"Invalid turn format in episode {episode_id}: {turn}")

            role = self._parse_role(turn.get("role", None))
            if role is None:
                raise DatasetError(f"Invalid or missing role in episode {episode_id}: {turn}")

            content = turn.get("content", None)
            if not content:
                raise DatasetError(f"Missing content in episode {episode_id}: {turn}")

            turn_metadata = dict(turn.get("metadata", {}))
            turn_metadata["turn_index"] = turn_idx

            chat_history.append(Message(role=role, content=str(content), metadata=turn_metadata))

        if not chat_history:
            msg = f"No valid turns found in episode {episode_id}"
            raise DatasetError(msg)

        return chat_history

    def _parse_role(self, role_str: str | None) -> Role | None:
        """Parse role string and return Role enum or None if invalid."""
        if role_str is None:
            return None
        role_lower = role_str.lower()
        if role_lower == "user":
            return Role.USER
        if role_lower == "assistant":
            return Role.ASSISTANT
        if role_lower == "system":
            return Role.SYSTEM
        return None

    def _build_metadata(self, record: Mapping[str, Any], split: str) -> dict[str, Any]:
        """Build episode metadata from record. Subclasses can override to add custom fields."""
        record_metadata = dict(record.get("metadata", {}))
        record_metadata.update(
            {
                "split": split,
                "dataset": record.get("dataset", self.default_dataset_name),
                "task_description": record.get("task_description"),
            }
        )
        # Not using this in the paper version due to observed instability
        record_metadata.pop("few_shot_user_examples", None)
        return record_metadata


__all__ = ["JSONLMirrorBase"]
