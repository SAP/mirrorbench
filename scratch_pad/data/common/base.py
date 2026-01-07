"""Base classes for dataset-specific preprocessors."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, MutableMapping, Sequence, Tuple

from . import get_logger, stratified_sample
from .embeddings import get_embedding_client
from .fewshot import FewShotRetriever
from .io import append_jsonl_record
from .task_description import TaskDescriptionBuilder
from .types import ConversationRecord, Turn


@dataclass(slots=True)
class CandidateConversation:
    conversation_id: str
    turns: Sequence[Turn]
    metadata: MutableMapping[str, object] = field(default_factory=dict)
    strat_key: Tuple[str, ...] = field(default_factory=tuple)
    explicit_hints: Sequence[str] = field(default_factory=tuple)


class BaseDatasetPreprocessor:
    dataset_name: str

    def __init__(
        self,
        *,
        output_dir: Path | str,
        max_samples: int = 500,
        seed: int = 13,
        task_builder: TaskDescriptionBuilder | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples
        self.rng = random.Random(seed)
        self.logger = get_logger(f"mirrorbench.preprocess.{self.dataset_name}")
        self.task_builder = task_builder or TaskDescriptionBuilder.create_default()

    def run(self, *, resume: bool = False) -> Path:
        output_path = self.output_dir / f"{self.dataset_name}_mirror.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        existing_count = 0
        existing_ids: set[str] = set()
        existing_records: list[dict[str, object]] = []
        if resume:
            existing_count, existing_ids, existing_records = self._load_existing_records(output_path)
            if existing_count:
                self.logger.info(
                    "Resuming preprocessing",
                    extra={"existing_records": existing_count, "path": str(output_path)},
                )
        else:
            if output_path.exists():
                output_path.unlink()

        remaining_budget = self.max_samples - existing_count
        if remaining_budget <= 0:
            self.logger.info(
                "Output already contains requested number of samples",
                extra={"max_samples": self.max_samples, "path": str(output_path)},
            )
            return output_path

        raw_total = 0
        candidates: list[CandidateConversation] = []
        for candidate in self.prepare_candidates():
            raw_total += 1
            if candidate.conversation_id in existing_ids:
                continue
            candidates.append(candidate)

        self.logger.info(
            "Prepared %d raw candidates (%d new)",
            raw_total,
            len(candidates),
        )

        if not candidates:
            self.logger.info("No new candidates available", extra={"path": str(output_path)})
            return output_path

        sampled = stratified_sample(
            candidates,
            stratify_fn=lambda c: c.strat_key,
            max_samples=min(remaining_budget, len(candidates)),
            rng=self.rng,
        )
        self.logger.info(
            "Selected %d stratified samples",
            len(sampled),
        )

        retriever = self._initialize_fewshot_retriever(existing_records)
        candidate_user_utterances: dict[str, list[str]] = {}
        for candidate in sampled:
            utterances = [turn.content for turn in self.iter_user_turns(candidate.turns) if turn.content.strip()]
            candidate_user_utterances[candidate.conversation_id] = utterances
            if retriever and utterances:
                retriever.add_examples(candidate.conversation_id, utterances)
        prepared = 0
        written = 0
        skipped = 0
        for candidate in sampled:
            try:
                prepared += 1
                task_description = self.build_task_description(candidate)
                record = ConversationRecord(
                    dataset=self.dataset_name,
                    conversation_id=candidate.conversation_id,
                    turns=list(candidate.turns),
                    task_description=task_description,
                    metadata=dict(candidate.metadata),
                )
                if retriever:
                    utterances = candidate_user_utterances.get(candidate.conversation_id, [])
                    examples = retriever.find_examples(
                        utterances,
                        exclude_conversation=candidate.conversation_id,
                    )
                    if examples:
                        record.metadata["few_shot_user_examples"] = examples
                append_jsonl_record(output_path, record)
                written += 1
                existing_ids.add(str(record.conversation_id))
            except Exception as exc:  # pragma: no cover - defensive logging
                skipped += 1
                if retriever:
                    retriever.remove_conversation(candidate.conversation_id)
                self.logger.warning(
                    "Skipping candidate due to processing error",
                    extra={
                        "conversation_id": candidate.conversation_id,
                        "error": str(exc),
                    },
                )

        total_records = existing_count + written
        self.logger.info(
            "Completed preprocessing",
            extra={
                "path": str(output_path),
                "written_this_run": written,
                "skipped": skipped,
                "total_records": total_records,
            },
        )
        return output_path

    def build_task_description(self, candidate: CandidateConversation) -> str:
        return self.task_builder.build(
            turns=candidate.turns,
            metadata=candidate.metadata,
            explicit_hints=candidate.explicit_hints,
            conversation_id=f"{self.dataset_name}:{candidate.conversation_id}",
        )

    def prepare_candidates(self) -> Iterable[CandidateConversation]:
        raise NotImplementedError

    def normalize_user_role(self, raw_role: str) -> str:
        role = raw_role.lower()
        if "assistant" in role or role in {"bot", "system"}:
            return "assistant"
        if role in {"human", "user", "customer", "querier", "prompter"}:
            return "user"
        return "user"

    def make_turn(self, *, role: str, content: str, metadata: Mapping[str, object] | None = None) -> Turn:
        return Turn(role=self.normalize_user_role(role), content=content.strip(), metadata=metadata or {})

    def iter_user_turns(self, turns: Sequence[Turn]) -> Iterator[Turn]:
        for turn in turns:
            if turn.role == "user":
                yield turn

    def _initialize_fewshot_retriever(
        self, existing_records: Sequence[Mapping[str, object]]
    ) -> FewShotRetriever | None:
        try:
            client = get_embedding_client()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning(f"Failed to initialise embedding client. Error: {exc}", extra={"error": str(exc)})
            return None

        if client is None:
            self.logger.info("Embedding client not configured; skipping few-shot retrieval")
            return None

        retriever = FewShotRetriever(client)
        for record in existing_records:
            conv_id = str(record.get("conversation_id", ""))
            if not conv_id:
                raise ValueError("Existing record missing conversation_id")
            turns = record.get("turns")
            if not isinstance(turns, list):
                continue
            utterances = (
                turn.get("content", "")
                for turn in turns
                if isinstance(turn, Mapping) and turn.get("role") == "user"
            )
            retriever.add_examples(conv_id, utterances)
        return retriever

    def _load_existing_records(self, path: Path) -> tuple[int, set[str], list[dict[str, object]]]:
        if not path.exists():
            return 0, set(), []
        existing_ids: set[str] = set()
        total = 0
        records: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    self.logger.warning(
                        "Skipping malformed JSONL line during resume",
                        extra={"path": str(path), "line_number": line_number, "error": str(exc)},
                    )
                    continue
                conversation_id = payload.get("conversation_id")
                if conversation_id is None:
                    continue
                existing_ids.add(str(conversation_id))
                records.append(payload)
                total += 1
        return total, existing_ids, records
