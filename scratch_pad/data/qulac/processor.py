from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, TYPE_CHECKING

from common.base import BaseDatasetPreprocessor, CandidateConversation
from common.io import download_file

if TYPE_CHECKING:
    from common.task_description import TaskDescriptionBuilder


class QulacPreprocessor(BaseDatasetPreprocessor):
    dataset_name = "qulac"
    DATA_URL = "https://raw.githubusercontent.com/aliannejadi/qulac/refs/heads/master/data/qulac/qulac.json"

    def __init__(
        self,
        *,
        output_dir: str,
        max_samples: int = 500,
        seed: int = 13,
        cache_dir: str | None = None,
        task_builder: "TaskDescriptionBuilder" | None = None,
    ) -> None:
        super().__init__(
            output_dir=output_dir,
            max_samples=max_samples,
            seed=seed,
            task_builder=task_builder,
        )
        self.cache_dir = Path(cache_dir) if cache_dir else Path(output_dir) / ".cached"

    def prepare_candidates(self) -> Iterable[CandidateConversation]:
        payload = self._load_dataset()
        indices = sorted(payload["question"].keys(), key=int)
        for idx in indices:
            conversation = self._build_conversation(payload, idx)
            if conversation is None:
                continue
            yield conversation

    def _load_dataset(self) -> Dict[str, Dict[str, str]]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.cache_dir / "qulac.json"
        download_file(self.DATA_URL, target_path)
        with target_path.open("r", encoding="utf-8") as f:
            data: Dict[str, Dict[str, str]] = json.load(f)
        return data

    def _build_conversation(
        self, payload: Dict[str, Dict[str, str]], index: str
    ) -> CandidateConversation | None:
        question = (payload.get("question", {}).get(index) or "").strip()
        answer = (payload.get("answer", {}).get(index) or "").strip()
        if not question or not answer:
            return None

        topic = (payload.get("topic", {}).get(index) or "").strip()
        topic_desc = (payload.get("topic_desc", {}).get(index) or "").strip()

        metadata: Dict[str, object] = {
            "topic": topic,
            "topic_id": payload.get("topic_id", {}).get(index),
            "facet_id": payload.get("facet_id", {}).get(index),
            "facet_desc": payload.get("facet_desc", {}).get(index),
            "topic_type": payload.get("topic_type", {}).get(index),
            "facet_type": payload.get("facet_type", {}).get(index),
            "language": "en",
            "topic_desc": (payload.get("topic_desc", {}).get(index) or "").strip(),
        }

        turns = [
            self.make_turn(role="assistant", content=question, metadata={"topic": topic}),
            self.make_turn(role="user", content=answer),
        ]

        strat_key = (
            str(metadata.get("topic", "unknown")),
            str(metadata.get("topic_type", "unknown")),
            str(metadata.get("facet_type", "unknown"))
        )

        explicit_hints: List[str] = []
        if topic_desc:
            explicit_hints.append(topic_desc)
        facet_desc = metadata.get("facet_desc")
        if isinstance(facet_desc, str) and facet_desc.strip():
            explicit_hints.append(facet_desc.strip())

        return CandidateConversation(
            conversation_id=f"qulac-{index}",
            turns=turns,
            metadata=metadata,
            strat_key=strat_key,
            explicit_hints=explicit_hints,
        )
