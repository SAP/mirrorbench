from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, TYPE_CHECKING

from datasets import load_dataset

from common.base import BaseDatasetPreprocessor, CandidateConversation
from common.types import Turn

if TYPE_CHECKING:
    from common.task_description import TaskDescriptionBuilder


class ChatbotArenaPreprocessor(BaseDatasetPreprocessor):
    dataset_name = "chatbot_arena"

    def __init__(
        self,
        *,
        output_dir: str,
        max_samples: int = 500,
        seed: int = 13,
        split: str = "train",
        task_builder: "TaskDescriptionBuilder" | None = None,
    ) -> None:
        super().__init__(
            output_dir=output_dir,
            max_samples=max_samples,
            seed=seed,
            task_builder=task_builder,
        )
        self.split = split

    def prepare_candidates(self) -> Iterable[CandidateConversation]:
        dataset = load_dataset("lmsys/chatbot_arena_conversations", split=self.split)
        for record in dataset:
            language_code = self._normalize_language(record.get("language"))
            if language_code and language_code != "en":
                continue
            if not language_code:
                language_code = "en"

            conversation, model_label = self._choose_conversation(record)
            if not conversation:
                continue

            turns = self._build_turns(conversation)
            if turns is None or turns[0].role != "user":
                continue
            if len(turns) < 2:
                continue
            user_turns = list(self.iter_user_turns(turns))
            if not user_turns:
                continue

            strat_key = (
                language_code,
                self._user_turn_bucket(len(user_turns)),
                "multi" if len(user_turns) > 1 else "single",
            )

            metadata: Dict[str, object] = {
                "question_id": record.get("question_id"),
                "arena_winner": record.get("winner"),
                "selected_model": model_label,
                "judge": record.get("judge"),
                "language": language_code,
                "turns_reported": record.get("turn"),
                "conversation_length": len(turns),
            }

            explicit_hints: List[str] = []

            yield CandidateConversation(
                conversation_id=f"{record.get('question_id')}::{metadata['selected_model']}",
                turns=turns,
                metadata=metadata,
                strat_key=strat_key,
                explicit_hints=explicit_hints,
            )

    def _choose_conversation(self, record: Dict[str, object]) -> tuple[Sequence[Dict[str, object]], str]:
        winner = record.get("winner")
        if winner == "model_a":
            return record.get("conversation_a", []), str(record.get("model_a"))
        if winner == "model_b":
            return record.get("conversation_b", []), str(record.get("model_b"))
        # fallback: take model_a conversation
        return record.get("conversation_a", []), str(record.get("model_a"))

    def _build_turns(self, conversation: Sequence[Dict[str, object]]) -> List[Turn]:
        turns: List[Turn] = []
        for item in conversation:
            content = str(item['content']).strip()
            role = str(item['role'])
            if not content:
                return None
            turns.append(self.make_turn(role=role, content=content))
        return turns

    @staticmethod
    def _user_turn_bucket(count: int) -> str:
        if count <= 1:
            return "user_turns:single"
        if count <= 3:
            return "user_turns:few"
        return "user_turns:many"

    @staticmethod
    def _normalize_language(value: object) -> str:
        raw = str(value or "").strip().lower()
        if not raw:
            return ""
        if raw == "english" or raw.startswith("en"):
            return "en"
        return raw
