"""Generic JSONL loader with optional filtering and field mapping."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from mirrorbench.datasets.loaders import register_loader
from mirrorbench.datasets.loaders.base import DatasetLoaderBackend, DatasetLoaderError


@register_loader("jsonl")
class JSONLLoader(DatasetLoaderBackend):
    """Load records from JSON Lines files stored on disk."""

    def _resolve_path(self, split: str) -> Path:
        raw_path = self.params.get("path")
        if raw_path is None:
            splits = self.params.get("splits")
            if not isinstance(splits, Mapping):
                msg = "JSONL loader requires either 'path' or 'splits' mapping"
                raise DatasetLoaderError(msg)
            try:
                raw_path = splits[split]
            except KeyError as exc:  # pragma: no cover - defensive path
                msg = f"No JSONL path configured for split '{split}'"
                raise DatasetLoaderError(msg) from exc
        candidate = Path(str(raw_path))
        if candidate.is_dir():
            candidate = candidate / f"{split}.jsonl"
        if not candidate.exists():
            msg = f"JSONL file '{candidate}' does not exist"
            raise DatasetLoaderError(msg)
        return candidate

    def load_split(
        self,
        *,
        split: str,
        limit: int | None = None,
    ) -> Iterable[Mapping[str, Any]]:
        file_path = self._resolve_path(split)
        encoding = str(self.params.get("encoding", "utf-8"))
        task_field = self.params.get("task_field")
        task_value = self.params.get("task_value")
        field_mapping: Mapping[str, str] = self.params.get("field_mapping", {})
        yielded = 0
        with file_path.open("r", encoding=encoding) as handle:
            for raw_line in handle:
                if limit is not None and yielded >= limit:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:  # pragma: no cover - defensive path
                    msg = f"Invalid JSON on line {yielded + 1} of '{file_path}'"
                    raise DatasetLoaderError(msg) from exc
                if task_field and task_value is not None and record.get(task_field) != task_value:
                    continue
                if field_mapping:
                    transformed = {dest: record.get(src) for dest, src in field_mapping.items()}
                else:
                    transformed = record
                yielded += 1
                yield transformed


__all__ = ["JSONLLoader"]
