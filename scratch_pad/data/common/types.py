"""Shared datatypes used during preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional


@dataclass(slots=True)
class Turn:
    """A single conversational turn."""

    role: Literal["user", "assistant"]
    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConversationRecord:
    """Normalized structure for a processed conversation."""

    dataset: str
    conversation_id: str
    turns: List[Turn]
    task_description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "conversation_id": self.conversation_id,
            "task_description": self.task_description,
            "turns": [
                {
                    "role": turn.role,
                    "content": turn.content,
                    "metadata": dict(turn.metadata),
                }
                for turn in self.turns
            ],
            "metadata": self.metadata,
        }
