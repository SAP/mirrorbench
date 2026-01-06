"""Messaging primitives shared across adapters, tasks, and judges."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class Role(str, Enum):
    """Enumerate speaker roles that can appear in a conversation transcript."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    JUDGE = "judge"


@dataclass(slots=True)
class Message:
    """One utterance in a conversation transcript."""

    role: Role
    """Speaker role."""

    content: str
    """Message content."""

    message_id: str = field(default_factory=lambda: str(uuid4()))
    """Stable identifier used to reference this message."""

    name: str | None = None
    """Optional speaker label exposed by some providers."""

    timestamp: datetime | None = None
    """When the message was generated. Defaults to ``None`` if unknown."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Provider-specific metadata (e.g., tool calls, annotations)."""


@dataclass(slots=True)
class TurnTelemetry:
    """Operational metrics captured for a single model invocation."""

    time_to_first_token: float | None = None
    """Latency between request submission and first token (seconds)."""

    time_per_output_token: float | None = None
    """Average time per generated token (seconds per token)."""

    total_response_time: float | None = None
    """Total wall-clock time for the turn (seconds)."""

    tokens_input: int | None = None
    """Number of tokens sent to the provider, if reported."""

    tokens_output: int | None = None
    """Number of tokens emitted by the provider, if reported."""

    cost_usd: float | None = None
    """Actual or estimated cost for the call in USD."""

    provider: str | None = None
    """Identifier for the model/provider handling the request."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional transport or tracing metadata (request ids, retries, etc.)."""


@dataclass(slots=True)
class JudgeVerdict:
    """Normalized representation of a judge model's assessment."""

    score: float
    """Primary score in the range dictated by the judge prompt (commonly [0, 1])."""

    label: str | None = None
    """Discrete label chosen by the judge (e.g., ``"pass"``/``"fail"``)."""

    confidence: float | None = None
    """Judge-reported confidence, when available."""

    reason: str | None = None
    """Natural-language explanation from the judge."""

    raw: dict[str, Any] = field(default_factory=dict)
    """Full raw payload returned by the judge invocation."""

    telemetry: TurnTelemetry | None = None
    """Telemetry associated with the judge call."""
