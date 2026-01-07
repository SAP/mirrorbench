"""Dataset orchestration utilities built on top of loader backends."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from mirrorbench.core.models.episodes import EpisodeSpec, ReferenceStats
from mirrorbench.core.models.messages import Message, Role
from mirrorbench.core.models.plan import DatasetSpec
from mirrorbench.core.registry import BaseDatasetLoader
from mirrorbench.datasets.artifacts import (
    ARTIFACT_MANIFEST_FILENAME,
    EPISODES_FILENAME,
    DatasetArtifact,
    compute_pipeline_hash,
)
from mirrorbench.datasets.loaders import DatasetLoaderBackend, DatasetLoaderError, create_loader
from mirrorbench.io.paths import Paths


class DatasetError(RuntimeError):
    """Raised when a dataset fails to materialise correctly."""


class BaseDataset(BaseDatasetLoader, ABC):
    """Common orchestration logic shared by dataset implementations."""

    loader_name: str
    """Name of the registered loader backend to use for this dataset."""

    variant: str = "episodes"
    """Variant of the dataset produced by this loader (e.g., 'episodes')."""

    pipeline_version: str = "1"
    """Version of the dataset pipeline (bump when changing pipeline logic)."""

    hash_ignored_params: frozenset[str] = frozenset({"refresh_cache"})
    """Dataset spec parameters to ignore when computing the pipeline hash."""

    def __init__(self, *, paths: Paths | None = None) -> None:
        self._paths = paths or Paths.default()

    # ------------------------------------------------------------------
    # Public API required by BaseDatasetLoader
    # ------------------------------------------------------------------
    def episodes(
        self,
        *,
        spec: DatasetSpec,
        split: str,
        limit: int | None = None,
    ) -> Iterable[EpisodeSpec]:
        artifact = self.materialize(spec=spec, split=split)
        episodes_path = artifact.path / EPISODES_FILENAME
        return self._iter_cached_episodes(episodes_path, limit=limit)

    def materialize(self, *, spec: DatasetSpec, split: str) -> DatasetArtifact:
        """Ensure the requested dataset split is cached and return its artifact."""

        loader_params = self.build_loader_params(spec)
        payload = self.build_pipeline_payload(
            spec=spec,
            split=split,
            loader_params=loader_params,
        )
        cache_key = compute_pipeline_hash(payload=payload)

        split_dir = self._paths.dataset_cache_dir(self.info.name, cache_key) / split
        episodes_path = split_dir / EPISODES_FILENAME
        manifest_path = split_dir / ARTIFACT_MANIFEST_FILENAME
        refresh = bool(spec.params.get("refresh_cache", False))
        if not refresh and episodes_path.exists() and manifest_path.exists():
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            return DatasetArtifact.from_manifest(manifest_data)

        split_dir.mkdir(parents=True, exist_ok=True)
        loader_limit = self.resolve_loader_limit(spec)
        episodes = list(
            self._generate_episodes(
                spec=spec,
                split=split,
                loader_params=loader_params,
                limit=loader_limit,
            )
        )
        self._write_episodes(episodes_path, episodes)
        artifact = DatasetArtifact(
            dataset_name=self.info.name,
            split=split,
            variant=self.variant,
            path=split_dir,
            cache_key=cache_key,
            metadata=self.build_artifact_metadata(
                spec=spec,
                split=split,
                episodes=episodes,
                loader_params=loader_params,
            ),
        )
        manifest_path.write_text(
            json.dumps(artifact.manifest(), indent=2),
            encoding="utf-8",
        )
        return artifact

    # ------------------------------------------------------------------
    # Extension hooks for subclasses
    # ------------------------------------------------------------------
    def build_loader_params(self, spec: DatasetSpec) -> Mapping[str, Any]:
        """Return parameters to pass to the loader backend."""

        loader_params = spec.params.get("loader", {})
        if not isinstance(loader_params, Mapping):
            msg = "Dataset 'loader' parameters must be a mapping"
            raise DatasetError(msg)
        return loader_params

    def resolve_loader_limit(self, spec: DatasetSpec) -> int | None:
        """Limit records fetched from the loader (e.g., for smoke tests)."""

        limit = spec.params.get("max_examples")
        return int(limit) if isinstance(limit, int) else None

    @abstractmethod
    def build_episode(
        self,
        *,
        record: Mapping[str, Any],
        spec: DatasetSpec,
        split: str,
        index: int,
    ) -> EpisodeSpec:
        """Convert a loader record into an :class:`EpisodeSpec`."""

    def build_pipeline_payload(
        self,
        *,
        spec: DatasetSpec,
        split: str,
        loader_params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return a mapping describing the pipeline for hashing."""

        dataset_params = {
            key: value for key, value in spec.params.items() if key not in self.hash_ignored_params
        }
        return {
            "dataset_name": self.info.name,
            "split": split,
            "pipeline_version": self.pipeline_version,
            "loader": {
                "name": self.loader_name,
                "params": loader_params,
            },
            "dataset_params": dataset_params,
        }

    def build_artifact_metadata(
        self,
        *,
        spec: DatasetSpec,
        split: str,
        episodes: list[EpisodeSpec],
        loader_params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Extra metadata stored alongside the cached artifact."""

        return {
            "episode_count": len(episodes),
            "split": split,
            "loader": {
                "name": self.loader_name,
                "params": dict(loader_params),
            },
            "dataset_params": {
                key: value
                for key, value in spec.params.items()
                if key not in self.hash_ignored_params
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_episodes(
        self,
        *,
        spec: DatasetSpec,
        split: str,
        loader_params: Mapping[str, Any],
        limit: int | None,
    ) -> Iterator[EpisodeSpec]:
        loader: DatasetLoaderBackend = create_loader(self.loader_name, params=loader_params)
        try:
            records = loader.load_split(split=split, limit=limit)
            for index, record in enumerate(records):
                yield self.build_episode(
                    record=record,
                    spec=spec,
                    split=split,
                    index=index,
                )
        except DatasetLoaderError as exc:
            raise DatasetError(str(exc)) from exc
        finally:
            loader.shutdown()

    def _write_episodes(self, path: Path, episodes: Iterable[EpisodeSpec]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for episode in episodes:
                payload = _episode_to_dict(episode)
                handle.write(json.dumps(payload, ensure_ascii=False))
                handle.write("\n")

    def _iter_cached_episodes(
        self,
        path: Path,
        *,
        limit: int | None,
    ) -> Iterator[EpisodeSpec]:
        read = 0
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if limit is not None and read >= limit:
                    break
                stripped = raw_line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                yield _episode_from_dict(payload)
                read += 1


def _episode_to_dict(episode: EpisodeSpec) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "task_tag": episode.task_tag,
        "chat_history": [_message_to_dict(msg) for msg in episode.chat_history],
        "references": episode.references,
        "metadata": episode.metadata,
        "reference_stats": _reference_stats_to_dict(episode.reference_stats),
    }


def _episode_from_dict(payload: Mapping[str, Any]) -> EpisodeSpec:
    return EpisodeSpec(
        episode_id=str(payload["episode_id"]),
        task_tag=str(payload["task_tag"]),
        chat_history=[_message_from_dict(item) for item in payload.get("chat_history", [])],
        references=dict(payload.get("references", {})),
        metadata=dict(payload.get("metadata", {})),
        reference_stats=_reference_stats_from_dict(payload.get("reference_stats")),
    )


def _message_to_dict(message: Message) -> dict[str, Any]:
    return {
        "role": message.role.value,
        "content": message.content,
        "message_id": message.message_id,
        "name": message.name,
        "timestamp": message.timestamp.isoformat() if message.timestamp else None,
        "metadata": message.metadata,
    }


def _message_from_dict(payload: Mapping[str, Any]) -> Message:
    timestamp = payload.get("timestamp")
    parsed_ts = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else None
    return Message(
        role=Role(str(payload["role"])),
        content=str(payload.get("content", "")),
        message_id=str(payload.get("message_id", "")),
        name=payload.get("name"),
        timestamp=parsed_ts,
        metadata=dict(payload.get("metadata", {})),
    )


def _reference_stats_to_dict(stats: ReferenceStats | None) -> dict[str, Any] | None:
    if stats is None:
        return None
    return {
        "schema_version": stats.schema_version,
        "metrics": stats.metrics,
        "distributions": stats.distributions,
        "generated_at": stats.generated_at.isoformat() if stats.generated_at else None,
    }


def _reference_stats_from_dict(payload: Any) -> ReferenceStats | None:
    if not payload:
        return None
    generated_at = payload.get("generated_at")
    timestamp = datetime.fromisoformat(generated_at) if isinstance(generated_at, str) else None
    return ReferenceStats(
        schema_version=str(payload.get("schema_version", "1.0")),
        metrics=dict(payload.get("metrics", {})),
        distributions=dict(payload.get("distributions", {})),
        generated_at=timestamp,
    )


__all__ = ["BaseDataset", "DatasetError"]
