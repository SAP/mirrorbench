from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, TYPE_CHECKING

from common.base import BaseDatasetPreprocessor, CandidateConversation
from common.io import download_file

if TYPE_CHECKING:
    from common.task_description import TaskDescriptionBuilder


class ClariQPreprocessor(BaseDatasetPreprocessor):
    dataset_name = "clariq"
    DATA_URL = "https://raw.githubusercontent.com/aliannejadi/ClariQ/master/data/multi_turn_human_generated_data.tsv"

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
        rows = self._load_rows()
        for idx, row in enumerate(rows):
            conversation = self._build_conversation(row, idx)
            if conversation is None:
                continue
            yield conversation

    def _load_rows(self) -> List[Dict[str, str]]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.cache_dir / "clariq.tsv"
        download_file(self.DATA_URL, target_path)
        with target_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            return list(reader)

    def _build_conversation(self, row: Dict[str, str], index: int) -> CandidateConversation | None:
        initial_request = (row.get("initial_request") or "").strip()
        if not initial_request:
            return None

        turns = [
            self.make_turn(
                role="user",
                content=initial_request,
                metadata={
                    "topic_id": row.get("topic_id"),
                    "facet_id": row.get("facet_id"),
                },
            )
        ]

        clarifications = 0
        for idx_turn in range(1, 4):
            question_key = f"question{idx_turn}"
            answer_key = f"answer{idx_turn}"
            question = (row.get(question_key) or "").strip()
            answer = (row.get(answer_key) or "").strip()
            if not question:
                continue
            turns.append(self.make_turn(role="assistant", content=question))
            if answer:
                turns.append(self.make_turn(role="user", content=answer))
                clarifications += 1

        if clarifications < 3:
            return None

        metadata: Dict[str, object] = {
            "topic_id": row.get("topic_id"),
            "facet_id": row.get("facet_id"),
            "facet": row.get("facet"),
            "clarification_pairs": clarifications,
            "language": "en",
        }

        strat_key = (
            f"topic_bucket:{self._topic_bucket(row.get('topic_id'))}",
        )

        explicit_hints: List[str] = []
        facet = row.get("facet")
        if facet:
            explicit_hints.append(facet.strip())

        return CandidateConversation(
            conversation_id=f"clariq-{index}",
            turns=turns,
            metadata=metadata,
            strat_key=strat_key,
            explicit_hints=explicit_hints,
        )

    @staticmethod
    def _topic_bucket(topic_id: str | None) -> str:
        if not topic_id:
            return "unknown"
        try:
            return str(int(topic_id) % 5)
        except ValueError:
            return topic_id[:2]

    @staticmethod
    def _turn_bucket(total_turns: int) -> str:
        if total_turns <= 3:
            return "turns:short"
        if total_turns <= 6:
            return "turns:medium"
        return "turns:long"
