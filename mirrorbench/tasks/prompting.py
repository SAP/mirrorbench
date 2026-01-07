"""Prompt construction helpers for task drivers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mirrorbench.core.models.episodes import EpisodeSpec
from mirrorbench.core.models.messages import Message

DEFAULT_MAX_REFERENCE_TURNS = 4


def build_user_proxy_system_prompt(
    *,
    episode: EpisodeSpec,
    dataset_name: str,
    dataset_label: str | None = None,
    max_reference_turns: int = DEFAULT_MAX_REFERENCE_TURNS,
) -> str:
    """Construct a system prompt guiding user-proxy models during evaluation.

    The prompt leverages any available metadata and reference conversations so user
    proxies can mirror the tone and behaviour of the target dataset.
    """

    task_description = _normalise_text(episode.metadata.get("task_description"))
    domain = _normalise_text(episode.metadata.get("domain"))
    persona = _normalise_text(episode.metadata.get("persona"))

    prompt_lines: list[str] = [
        "You are simulating a real human user for the MirrorBench evaluation harness.",
        "Respond with the next USER turn only. Do not write assistant messages, notes, or any other analysis.",
        "Your utterance should be like a real user and the context should be based on the following information provided.",
    ]

    if task_description:
        prompt_lines.append(f"Task description: {task_description}.")
    if domain:
        prompt_lines.append(f"Domain or topic: {domain}.")
    if persona:
        prompt_lines.append(f"Persona hints: {persona}.")
    if not any([task_description, domain, persona]):
        prompt_lines.append(
            "No additional dataset metadata provided. Respond naturally and plausibly based on the ongoing conversation."
        )

    prompt_lines.append(
        "Match the length, tone, and specificity of real user utterances. If you are unsure, "
        "respond naturally based on the assistant's previous messages like how a real human would. "
        "Note that your response MUST not contain anything other than the USER utterance. Do not "
        "include any prefixes like 'User:' or 'Human:' as well. Just the raw message content."
    )

    return "\n".join(prompt_lines)


def _normalise_text(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    return text or None


def _format_conversation(messages: Sequence[Message]) -> str:
    """Render a sequence of messages into a deterministic transcript."""

    lines: list[str] = []
    for message in messages:
        lines.append(f"{message.role.value.upper()}: {message.content}")
    return "\n".join(lines)


def build_assistant_mirror_system_prompt(
    *,
    episode: EpisodeSpec,
    dataset_name: str,
    dataset_label: str | None = None,
) -> str:
    """Construct the system prompt for assistant replicas in mirror evaluations."""
    real_conversation = _format_conversation(episode.chat_history)
    if len(real_conversation) == 0:
        raise ValueError("Cannot build assistant mirror prompt: no real conversation available")
    prompt = f"""
You are the assistant in a MirrorBench replay. The user-proxy agent is attempting to
reproduce the USER side of the real conversation provided below. But the user-proxy
does not have access to the real conversation hitory. Instead, it only has access to
the conversation summary.

You need to respond as the assistant. But we are providing you with the real
conversation history as context, so you can respond consistently same as (or similar to)
the original assistant in the real conversation (you may paraphrase lightly for safety).

If user-proxy deviates from the original USER turn or the original response would violate
policy, reply helpfully using your own knowledge while remaining consistent with the
persona demonstrated so far. Always follow Azure OpenAI content policies. Paraphrase sensitive content
instead of quoting it verbatim, and refuse politely if a request is disallowed.

Here is the real conversation for context (the USER turns are from the original conversation):
{real_conversation}

Now, we will provide you with ongoing conversation with the user-proxy. Please respond
as the assistant in this conversation.
""".strip()
    return prompt


__all__ = [
    "build_user_proxy_system_prompt",
    "build_assistant_mirror_system_prompt",
]
