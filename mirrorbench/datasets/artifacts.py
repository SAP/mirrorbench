"""Dataset artifact metadata helpers and pipeline hashing utilities."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

ARTIFACT_MANIFEST_FILENAME = "artifact.json"
EPISODES_FILENAME = "episodes.jsonl"


@dataclass(slots=True, frozen=True)
class DatasetArtifact:
    """Describe a materialized view of a dataset split."""

    dataset_name: str
    """Name of the dataset."""

    split: str
    """Name of the dataset split (e.g. "train", "test")."""

    variant: str
    """Variant of the dataset (e.g. "v1", "v2")."""

    path: Path
    """Path to the root of the dataset artifact."""

    cache_key: str
    """Unique cache key for this artifact."""

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """Timestamp when the artifact was created."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Optional additional metadata about the artifact."""

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-serialisable manifest for this artifact."""

        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "variant": self.variant,
            "path": str(self.path),
            "cache_key": self.cache_key,
            "created_at": self.created_at.isoformat(),
            "metadata": _canonicalise(self.metadata),
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> DatasetArtifact:
        """Rebuild an artifact from a manifest mapping."""

        created_at = payload.get("created_at")
        timestamp = (
            datetime.fromisoformat(created_at) if isinstance(created_at, str) else datetime.now(UTC)
        )
        return cls(
            dataset_name=str(payload["dataset_name"]),
            split=str(payload["split"]),
            variant=str(payload["variant"]),
            path=Path(str(payload["path"])),
            cache_key=str(payload["cache_key"]),
            created_at=timestamp,
            metadata=payload.get("metadata", {}),
        )


def compute_pipeline_hash(*, payload: Mapping[str, Any]) -> str:
    """Compute a deterministic hash for a dataset transform pipeline."""

    canonical = _canonicalise(payload)
    normalised = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()
    return digest[:16]


def _canonicalise(value: Any) -> Any:  # noqa: PLR0911
    """Convert arbitrary values into a JSON-serialisable, deterministic form."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonicalise(dataclasses.asdict(value))
    if isinstance(value, BaseModel):
        return _canonicalise(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {
            str(k): _canonicalise(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list | tuple)):
        return [_canonicalise(v) for v in value]
    if isinstance(value, (set | frozenset)):
        return sorted(_canonicalise(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


# ruff: noqa: RUF022
__all__ = [
    "ARTIFACT_MANIFEST_FILENAME",
    "DatasetArtifact",
    "EPISODES_FILENAME",
    "compute_pipeline_hash",
]
