"""Utilities for working with dataset and proxy conversations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from mirrorbench.core.models.episodes import EpisodeArtifact
from mirrorbench.core.models.messages import Message, Role

DEFAULT_REFERENCE_KEYS: tuple[str, ...] = (
    "real_conversation",
    "dataset_conversation",
    "reference_conversation",
)


def resolve_reference_conversation(
    episode: EpisodeArtifact,
    *,
    preferred_keys: Sequence[str] = DEFAULT_REFERENCE_KEYS,
    fallback_to_chat_history: bool = True,
) -> list[Message]:
    """Resolve the dataset conversation stored on an episode.

    Parameters
    ----------
    episode:
        Episode artefact containing references and chat history.
    preferred_keys:
        Ordered keys to inspect within ``episode.spec.references`` to locate the
        conversation. Defaults to ``(DEFAULT_REFERENCE_KEYS)``.
    fallback_to_chat_history:
        When ``True`` (default), return ``episode.spec.chat_history`` if no
        reference key yields a conversation.

    Returns
    -------
    list[Message]
        Normalised conversation as :class:`Message` instances.

    Raises
    ------
    ValueError
        If the references contain an unsupported structure (e.g., missing
        ``role``/``content`` keys or non-iterable types) or if no conversation
        is available and ``fallback_to_chat_history`` is ``False``.
    """

    reference_map = episode.spec.references or {}

    candidate: Any = None
    for key in preferred_keys:
        candidate = reference_map.get(key)
        if candidate:
            break

    if candidate is None and fallback_to_chat_history:
        candidate = episode.spec.chat_history

    if not candidate:
        return []

    if isinstance(candidate, list) and all(isinstance(msg, Message) for msg in candidate):
        return list(candidate)

    if isinstance(candidate, Iterable):
        resolved: list[Message] = []
        for item in candidate:
            if isinstance(item, Message):
                resolved.append(item)
                continue
            if isinstance(item, dict):
                role_value = item.get("role")
                content = item.get("content")
                if role_value is None or content is None:
                    msg = "Reference conversation entry must include 'role' and 'content' fields"
                    raise ValueError(msg)
                role = role_value if isinstance(role_value, Role) else Role(role_value)
                resolved.append(Message(role=role, content=str(content)))
                continue
            msg = (
                "Reference conversation entries must be Message instances or dictionaries"
                " with 'role'/'content' keys"
            )
            raise ValueError(msg)
        return resolved

    msg = "Unsupported reference conversation format"
    raise ValueError(msg)


__all__ = [
    "DEFAULT_REFERENCE_KEYS",
    "resolve_reference_conversation",
]
