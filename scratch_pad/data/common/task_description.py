"""Utilities for synthesizing task descriptions from conversations."""

from __future__ import annotations

import json
import os
import textwrap
from typing import Iterable, Mapping, MutableMapping, Protocol, Sequence

from mirrorbench.core.models.messages import Message, Role
from mirrorbench.model_clients.langchain import LangChainChatClient
from mirrorbench.model_clients.base import ChatResponse
from mirrorbench.model_clients.exceptions import ModelClientError

from .logging import get_logger
from .types import Turn


class _ChatInvoker(Protocol):
    def invoke(self, *, messages: Sequence[Message], **kwargs) -> ChatResponse: ...


DEFAULT_SYSTEM_PROMPT = (
    "You convert conversation analyses into concise task descriptions for a user-proxy "
    "agent. Summaries must capture the user's overarching objective, desired outcome, "
    "and conversational tone while omitting verbatim quotes. Output two to four "
    "sentences of natural language with no leading labels or bullet points."
)


class TaskDescriptionBuilder:
    """Task description generator backed by an LLM."""

    DEFAULT_CLIENT_NAME = "client:langchain/chat"
    DEFAULT_MODEL_IMPORT = "langchain_openai.AzureChatOpenAI"

    def __init__(
        self,
        llm_client: _ChatInvoker | None,
        *,
        max_hint_turns: int = 3,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        invoke_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.max_hint_turns = max_hint_turns
        self.system_prompt = system_prompt
        self.invoke_kwargs = dict(invoke_kwargs or {"temperature": 0})
        self.logger = get_logger("mirrorbench.preprocess.task_description")
        self._cache: MutableMapping[str, str] = {}

    @classmethod
    def create_default(
        cls,
        *,
        client_name: str | None = None,
        model_import: str | None = None,
        model_kwargs: Mapping[str, object] | None = None,
        invoke_kwargs: Mapping[str, object] | None = None,
        max_hint_turns: int = 3,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> "TaskDescriptionBuilder":
        from mirrorbench.core.registry import registry

        resolved_client = client_name or os.getenv(
            "MIRRORBENCH_TASK_DESC_CLIENT", cls.DEFAULT_CLIENT_NAME
        )
        resolved_model_import = model_import or os.getenv(
            "MIRRORBENCH_TASK_DESC_MODEL_IMPORT", cls.DEFAULT_MODEL_IMPORT
        )

        model_kwargs_payload = model_kwargs
        if model_kwargs_payload is None:
            raw_kwargs = os.getenv("MIRRORBENCH_TASK_DESC_MODEL_KWARGS")
            if raw_kwargs:
                try:
                    parsed = json.loads(raw_kwargs)
                except json.JSONDecodeError as exc:
                    raise ValueError("Invalid JSON in MIRRORBENCH_TASK_DESC_MODEL_KWARGS") from exc
                if not isinstance(parsed, Mapping):
                    raise ValueError("Model kwargs env var must decode to a mapping")
                model_kwargs_payload = parsed

        client_cls = registry.factory("model_clients", resolved_client)
        llm_client = client_cls(
            model_import=resolved_model_import,
            model_kwargs=dict(model_kwargs_payload or {}),
        )

        return cls(
            llm_client,
            max_hint_turns=max_hint_turns,
            system_prompt=system_prompt,
            invoke_kwargs=invoke_kwargs,
        )

    def build(
        self,
        *,
        turns: Sequence[Turn],
        metadata: Mapping[str, object] | None = None,
        explicit_hints: Iterable[str] | None = None,
        conversation_id: str | None = None,
    ) -> str:
        cache_key = conversation_id
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]

        prompt = self._build_prompt(turns=turns, metadata=metadata, explicit_hints=explicit_hints)

        if self.llm_client is None:
            raise RuntimeError("TaskDescriptionBuilder requires an LLM client")

        description: str | None = None

        if self.llm_client is not None:
            messages = [
                Message(role=Role.SYSTEM, content=self.system_prompt),
                Message(role=Role.USER, content=prompt),
            ]
            try:
                response = self.llm_client.invoke(messages=messages, **self.invoke_kwargs)
                description = (response.message.content or "").strip()
            except (ModelClientError, Exception) as exc:  # pragma: no cover - defensive
                self.logger.error(
                    "Task description LLM invocation failed", extra={"error": str(exc)}
                )
                raise RuntimeError("Failed to synthesize task description via LLM") from exc

        if not description:
            raise RuntimeError("Task description LLM returned an empty response")

        if cache_key:
            self._cache[cache_key] = description
        return description

    def _build_prompt(
        self,
        *,
        turns: Sequence[Turn],
        metadata: Mapping[str, object] | None,
        explicit_hints: Iterable[str] | None,
    ) -> str:
        metadata_block = self._format_metadata(metadata)
        conversation_outline = self._format_conversation_outline(turns)
        hint_block = self._format_explicit_hints(explicit_hints)
        followup = self._collect_followup_turns(turns)
        followup_block = " | ".join(followup) if followup else "None"

        body = textwrap.dedent(
            f"""
            We are preparing guidance for a user-proxy agent that must imitate the human user from an existing conversation.
            The actual transcript is provided for your analysis, but the resulting description must not quote or closely paraphrase any of the turns.
            Explain the conversation goal, high-level plan, and user persona in two to four sentences.

            Conversation metadata:
            {metadata_block}

            Conversation outline (internal reference only):
            {conversation_outline}

            Notable user checkpoints:
            {followup_block}

            Additional cues to preserve:
            {hint_block}

            Produce only the final description text with no list formatting, headings, or explanations of your reasoning.
            """
        ).strip()
        return body

    def _format_metadata(self, metadata: Mapping[str, object] | None) -> str:
        if not metadata:
            return "<none provided>"
        lines: list[str] = []
        for key, value in sorted(metadata.items()):
            if isinstance(value, (str, int, float, bool)):
                lines.append(f"- {key}: {self._compact_text(str(value), max_length=160)}")
        return "\n".join(lines) if lines else "<none provided>"

    def _format_conversation_outline(self, turns: Sequence[Turn]) -> str:
        lines: list[str] = []
        for idx, turn in enumerate(turns, start=1):
            content = self._compact_text(turn.content, max_length=240)
            lines.append(f"{turn.role.capitalize()} turn {idx}: {content}")
        return "\n".join(lines) if lines else "<no turns>"

    def _format_explicit_hints(self, explicit_hints: Iterable[str] | None) -> str:
        if not explicit_hints:
            return "<none>"
        hints = [self._compact_text(hint, max_length=160) for hint in explicit_hints if hint]
        if not hints:
            return "<none>"
        return " | ".join(hints[: self.max_hint_turns])

    def _collect_followup_turns(self, turns: Sequence[Turn]) -> list[str]:
        hints: list[str] = []
        for turn in turns:
            if turn.role.lower() != "user":
                continue
            compact = self._compact_text(turn.content, max_length=160)
            if compact:
                hints.append(compact)
            if len(hints) >= self.max_hint_turns:
                break
        return hints

    @staticmethod
    def _compact_text(text: str, *, max_length: int = 160) -> str:
        collapsed = " ".join(text.strip().split())
        if len(collapsed) <= max_length:
            return collapsed
        return collapsed[: max_length - 1].rstrip() + "…"
