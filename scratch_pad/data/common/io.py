"""IO utilities for preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import requests

from .types import ConversationRecord


def write_jsonl(path: str | Path, records: Iterable[ConversationRecord | Mapping[str, object]]) -> None:
    """Write records to a JSONL file, expanding dataclass instances when present."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            if isinstance(record, ConversationRecord):
                payload = record.to_json()
            else:
                payload = record
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def download_file(url: str, target_path: str | Path, *, force: bool = False) -> Path:
    """Download a remote file if it does not already exist."""
    path = Path(target_path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def append_jsonl_record(path: str | Path, record: ConversationRecord | Mapping[str, object]) -> None:
    """Append a single record to the specified JSONL file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(record, ConversationRecord):
        payload = record.to_json()
    else:
        payload = record
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
