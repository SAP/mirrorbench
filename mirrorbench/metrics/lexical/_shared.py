"""Shared helpers for lexical diversity metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from statistics import mean, stdev

from mirrorbench.core.models.episodes import EpisodeArtifact
from mirrorbench.core.models.messages import Message, Role
from mirrorbench.metrics.util.conversation import resolve_reference_conversation
from mirrorbench.metrics.util.text import tokenize


def messages_for_role(messages: Iterable[Message], role: Role) -> list[str]:
    """Return contents for messages that match the requested role."""

    return [message.content for message in messages if message.role == role]


def tokens_for_role(messages: Iterable[Message], role: Role, *, model: str) -> list[int]:
    """Tokenize concatenated text for the given role using tiktoken."""

    texts = messages_for_role(messages, role)
    if not texts:
        return []
    joined = " ".join(texts)
    return tokenize(joined, model=model)


def proxy_tokens(episode: EpisodeArtifact, role: Role, *, model: str) -> list[int]:
    """Collect tokens for proxy-generated turns with the specified role."""

    return tokens_for_role(episode.turns, role, model=model)


def human_tokens(episode: EpisodeArtifact, role: Role, *, model: str) -> list[int]:
    """Collect tokens for the real conversation stored alongside the episode."""

    reference_messages = resolve_reference_conversation(episode)
    if reference_messages:
        return tokens_for_role(reference_messages, role, model=model)
    return tokens_for_role(episode.spec.chat_history, role, model=model)


def mean_and_stdev(values: Sequence[float]) -> tuple[float, float]:
    """Return mean and sample standard deviation, guarding small samples."""

    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


__all__ = [
    "human_tokens",
    "mean_and_stdev",
    "messages_for_role",
    "proxy_tokens",
    "tokens_for_role",
]
