from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, MutableMapping, Sequence, TYPE_CHECKING

from datasets import load_dataset

from common.base import BaseDatasetPreprocessor, CandidateConversation
from common.types import Turn

if TYPE_CHECKING:
    from common.task_description import TaskDescriptionBuilder


class OASST1Preprocessor(BaseDatasetPreprocessor):
    dataset_name = "oasst1"

    def __init__(
        self,
        *,
        output_dir: str,
        max_samples: int = 500,
        seed: int = 13,
        split: str = "train+validation",
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
        dataset = load_dataset("OpenAssistant/oasst1", split=self.split)
        trees: Dict[str, List[MutableMapping[str, object]]] = defaultdict(list)
        for record in dataset:
            if record.get("deleted"):
                continue
            if record.get("synthetic"):
                continue
            text = (record.get("text") or "").strip()
            if not text:
                continue
            trees[str(record["message_tree_id"])].append(record)

        for tree_id, messages in trees.items():
            root = self._find_root(messages)
            if root is None:
                continue
            children_map = self._build_children_map(messages)
            language_code = self._normalize_language(root.get("lang"))
            if language_code != "en":
                continue
            turns = self._build_linear_path(root, children_map)
            if len(turns) < 2:
                continue
            user_turns = list(self.iter_user_turns(turns))
            if not user_turns:
                continue
            metadata: Dict[str, object] = {
                "language": language_code,
                "tree_state": root.get("tree_state", ""),
                "review_result": root.get("review_result", False),
                "dataset_split": self.split,
                "num_turns": len(turns),
                "num_user_turns": len(user_turns),
            }
            labels = self._extract_labels(root)
            if labels:
                metadata["labels"] = labels

            strat_key = (
                self._turn_bucket(len(user_turns)),
            )

            yield CandidateConversation(
                conversation_id=tree_id,
                turns=turns,
                metadata=metadata,
                strat_key=strat_key,
                explicit_hints=labels,
            )

    def _find_root(self, messages: Sequence[MutableMapping[str, object]]) -> MutableMapping[str, object] | None:
        for message in messages:
            if message.get("parent_id") is None and message.get("role") == "prompter":
                return message
        return None

    def _build_children_map(
        self, messages: Sequence[MutableMapping[str, object]]
    ) -> Dict[str, List[MutableMapping[str, object]]]:
        children: Dict[str, List[MutableMapping[str, object]]] = defaultdict(list)
        for message in messages:
            parent_id = message.get("parent_id")
            if parent_id is None:
                continue
            children[str(parent_id)].append(message)
        return children

    def _build_linear_path(
        self,
        root: MutableMapping[str, object],
        children_map: Dict[str, List[MutableMapping[str, object]]],
    ) -> List[Turn]:
        turns: List[Turn] = []
        visited: set[str] = set()
        current = root
        previous_role = ""

        while current is not None:
            message_id = str(current["message_id"])
            if message_id in visited:
                break
            visited.add(message_id)

            role = str(current['role'])
            if role not in {"prompter", "assistant"}:
                break

            normalized_role = "user" if role == "prompter" else "assistant"
            if normalized_role == previous_role:
                break

            turn_metadata = {
                "lang": current.get("lang"),
                "rank": current.get("rank"),
                "message_id": message_id,
            }
            turns.append(
                self.make_turn(
                    role=normalized_role,
                    content=str(current['text']),
                    metadata=turn_metadata,
                )
            )
            previous_role = normalized_role

            children = [
                child
                for child in children_map.get(message_id, [])
                if not child.get("deleted")
                and not child.get("synthetic")
                and str(child.get("text", "")).strip()
            ]
            if not children:
                break
            children.sort(key=self._child_sort_key)
            next_child = None
            for candidate in children:
                candidate_role = str(candidate['role'])
                if candidate_role in {"prompter", "assistant"}:
                    if (candidate_role == "prompter" and normalized_role == "assistant") or (
                        candidate_role == "assistant" and normalized_role == "user"
                    ):
                        next_child = candidate
                        break
            if next_child is None:
                break
            current = next_child

        return turns

    @staticmethod
    def _child_sort_key(message: MutableMapping[str, object]) -> tuple[int, str]:
        rank = message.get("rank")
        rank_value = int(rank) if isinstance(rank, int) else 1_000_000
        created = str(message.get("created_date") or "")
        return (rank_value, created)

    @staticmethod
    def _extract_labels(message: MutableMapping[str, object]) -> List[str]:
        labels_obj = message.get("labels")
        if not isinstance(labels_obj, MutableMapping):
            return []
        names = labels_obj.get("name")
        values = labels_obj.get("value")
        if not isinstance(names, Sequence) or not isinstance(values, Sequence):
            return []
        results: List[str] = []
        for name, value in zip(names, values):
            if value is None:
                continue
            if float(value) >= 0.7:
                results.append(f"{name}:{float(value):.2f}")
        return results[:3]

    @staticmethod
    def _turn_bucket(user_turns: int) -> str:
        if user_turns <= 1:
            return "user_turns:short"
        if user_turns <= 3:
            return "user_turns:medium"
        return "user_turns:long"

    @staticmethod
    def _normalize_language(value: object) -> str:
        raw = str(value or "").strip().lower()
        if not raw:
            return ""
        if raw == "english" or raw.startswith("en"):
            return "en"
        return raw
