from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import Iterable, Sequence

from .embeddings import EmbeddingClient


@dataclass
class _IndexedUtterance:
    conversation_id: str
    text: str
    embedding: Sequence[float]
    norm: float


class FewShotRetriever:
    """In-memory cosine-similarity retriever for user utterances."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        *,
        rng: Random | None = None,
        diversity_threshold: float = 0.70,
    ) -> None:
        self._client = embedding_client
        self._entries: list[_IndexedUtterance] = []
        self._rng = rng or Random()
        if not 0 < diversity_threshold < 1:
            raise ValueError("diversity_threshold must be between 0 and 1")
        self._diversity_threshold = diversity_threshold

    def add_examples(self, conversation_id: str, utterances: Iterable[str]) -> None:
        _texts = [text for text in utterances if text.strip()]
        texts = []
        for text in _texts:
            if any(entry.text == text and entry.conversation_id == conversation_id for entry in self._entries):
                continue
            texts.append(text)
        if not texts:
            return
        

        embeddings = self._client.embed(texts)
        for text, vector in zip(texts, embeddings):
            norm = _l2_norm(vector)
            if norm == 0:
                continue
            self._entries.append(
                _IndexedUtterance(
                    conversation_id=conversation_id,
                    text=text,
                    embedding=tuple(vector),
                    norm=norm,
                )
            )

    def remove_conversation(self, conversation_id: str) -> None:
        if not self._entries:
            return
        self._entries = [entry for entry in self._entries if entry.conversation_id != conversation_id]

    def find_examples(
        self,
        query_utterances: Iterable[str],
        *,
        exclude_conversation: str,
        top_k: int = 5,
    ) -> list[dict[str, str]]:
        queries = [text for text in query_utterances if text.strip()]
        if not queries or not self._entries:
            return []

        query_vectors = self._client.embed(queries)
        candidates_by_query: list[list[tuple[float, _IndexedUtterance]]] = []

        for query_vector in query_vectors:
            q_norm = _l2_norm(query_vector)
            if q_norm == 0:
                continue
            per_query: list[tuple[float, _IndexedUtterance]] = []
            for entry in self._entries:
                if entry.conversation_id == exclude_conversation:
                    continue
                similarity = _cosine_similarity(query_vector, q_norm, entry)
                per_query.append((similarity, entry))
            if not per_query:
                continue
            per_query.sort(key=lambda item: item[0], reverse=True)
            candidates_by_query.append(per_query[:(top_k * 2)])

        if not candidates_by_query:
            return []

        selected: list[dict[str, str]] = []
        selected_entries: list[_IndexedUtterance] = []
        used: set[tuple[str, str]] = set()

        while len(selected) < top_k:
            active_indices = [idx for idx, entries in enumerate(candidates_by_query) if entries]
            if not active_indices:
                break
            chosen_index = self._rng.choice(active_indices)
            similarity, entry = candidates_by_query[chosen_index].pop(0)
            key = (entry.conversation_id, entry.text)
            if key in used:
                continue
            if selected_entries:
                max_sim = max(_entry_cosine_similarity(entry, existing) for existing in selected_entries)
                if max_sim >= self._diversity_threshold:
                    continue
            used.add(key)
            selected_entries.append(entry)
            selected.append(
                {
                    "conversation_id": entry.conversation_id,
                    "utterance": entry.text,
                    "similarity": f"{similarity:.4f}",
                }
            )

        return selected


def _cosine_similarity(
    query_vector: Sequence[float], query_norm: float, entry: _IndexedUtterance
) -> float:
    denom = query_norm * entry.norm
    if denom == 0:
        raise ValueError(f"Zero norm encountered in cosine similarity calculation for entry: {entry.conversation_id}")
    return _dot_product(query_vector, entry.embedding) / denom


def _dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(l * r for l, r in zip(left, right))


def _l2_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _entry_cosine_similarity(left: _IndexedUtterance, right: _IndexedUtterance) -> float:
    denom = left.norm * right.norm
    if denom == 0:
        return 0.0
    return _dot_product(left.embedding, right.embedding) / denom
