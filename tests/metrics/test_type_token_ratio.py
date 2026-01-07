"""Unit tests for TypeTokenRatioMetric."""

from __future__ import annotations

from itertools import chain

import pytest

from mirrorbench.core.models.episodes import EpisodeArtifact, EpisodeSpec
from mirrorbench.core.models.messages import Message, Role
from mirrorbench.core.models.run import MetricValue
from mirrorbench.metrics.lexical.type_token_ratio import (
    TTR_METRIC_INFO,
    TypeTokenRatioMetric,
)
from mirrorbench.metrics.util.text import tokenize

METRIC_NAME = "metric:lexical/ttr"


def _make_episode(
    *,
    episode_id: str,
    user_turns: list[str],
    assistant_turns: list[str] | None = None,
) -> EpisodeArtifact:
    """Create a synthetic episode artifact for testing.

    Args:
        episode_id: Unique episode identifier.
        user_turns: List of user turn contents.
        assistant_turns: Optional list of assistant turn contents (interleaved with user).

    Returns:
        EpisodeArtifact with alternating user/assistant messages.
    """
    spec = EpisodeSpec(
        episode_id=episode_id,
        task_tag="test_task",
        chat_history=[],
    )

    turns: list[Message] = []
    for i, user_content in enumerate(user_turns):
        turns.append(Message(role=Role.USER, content=user_content))
        if assistant_turns and i < len(assistant_turns):
            turns.append(Message(role=Role.ASSISTANT, content=assistant_turns[i]))

    return EpisodeArtifact(spec=spec, turns=turns)


def _corpus_counts(values: list[MetricValue]) -> tuple[int, int]:
    """Aggregate token counts across metric values."""

    all_tokens = list(chain.from_iterable(value.metadata.get("token_ids", []) for value in values))
    total = len(all_tokens)
    unique = len(set(all_tokens))
    return total, unique


def test_ttr_metric_info() -> None:
    """Verify TTR metric metadata is correctly defined."""
    assert TTR_METRIC_INFO.name == METRIC_NAME
    assert TTR_METRIC_INFO.needs_references is False
    assert TTR_METRIC_INFO.needs_judge is False
    assert TTR_METRIC_INFO.category == "lexical_diversity"


def test_ttr_metric_single_episode_basic() -> None:
    """Test TTR evaluation on a single episode with simple text."""
    metric = TypeTokenRatioMetric(min_tokens=6)

    episode = _make_episode(
        episode_id="ep1",
        user_turns=["hello world hello again hello"],
        assistant_turns=["hi there"],
    )

    value = metric.evaluate(episode)

    token_ids = value.metadata["token_ids"]
    target_text = " ".join(msg.content for msg in episode.turns if msg.role == Role.USER)
    expected_tokens = tokenize(target_text, model=metric.tokenizer_model)

    assert value.metric_name == METRIC_NAME
    assert value.values == []
    assert token_ids == expected_tokens
    assert value.metadata["token_count"] == len(token_ids)
    assert value.metadata["below_threshold"] is (len(token_ids) < metric.min_tokens)
    assert value.metadata["target_role"] == Role.USER.value
    assert episode.metric_values[METRIC_NAME] is value


def test_ttr_metric_single_episode_above_threshold() -> None:
    """Test TTR evaluation on a episode above min_tokens threshold."""
    metric = TypeTokenRatioMetric(min_tokens=3)

    episode = _make_episode(
        episode_id="ep2",
        user_turns=["the quick brown fox jumps over the lazy dog"],
    )

    value = metric.evaluate(episode)

    token_ids = value.metadata["token_ids"]
    target_text = " ".join(msg.content for msg in episode.turns if msg.role == Role.USER)
    expected_tokens = tokenize(target_text, model=metric.tokenizer_model)

    assert token_ids == expected_tokens
    assert value.metadata["token_count"] == len(token_ids) > 0
    assert value.metadata["below_threshold"] is False
    assert value.metadata["min_tokens"] == metric.min_tokens
    assert len(set(token_ids)) <= len(token_ids)
    assert episode.metric_values[METRIC_NAME] is value


def test_ttr_metric_multiple_user_turns() -> None:
    """Test TTR with multiple user turns in same episode."""
    metric = TypeTokenRatioMetric()

    episode = _make_episode(
        episode_id="ep3",
        user_turns=[
            "hello there",
            "how are you doing today",
        ],
        assistant_turns=["hi"],
    )

    value = metric.evaluate(episode)

    target_text = " ".join(msg.content for msg in episode.turns if msg.role == Role.USER)
    expected_tokens = tokenize(target_text, model=metric.tokenizer_model)

    assert value.metadata["token_ids"] == expected_tokens
    assert value.metadata["token_count"] == len(expected_tokens)
    assert value.metadata["target_role"] == Role.USER.value


def test_ttr_metric_accepts_string_role() -> None:
    """Ensure the metric accepts role strings when instantiated via configuration."""
    metric = TypeTokenRatioMetric(target_role="assistant")

    episode = _make_episode(
        episode_id="ep-role",
        user_turns=["ignore me"],
        assistant_turns=["assistant response"],
    )

    value = metric.evaluate(episode)

    assert metric.target_role == Role.ASSISTANT
    assert value.metadata["target_role"] == Role.ASSISTANT.value


def test_ttr_metric_ignores_assistant_turns() -> None:
    """Verify metric only counts USER role messages."""
    metric = TypeTokenRatioMetric()

    episode = _make_episode(
        episode_id="ep4",
        user_turns=["hello world"],
        assistant_turns=["this is a long assistant response with many words"],
    )

    value = metric.evaluate(episode)

    user_text = " ".join(msg.content for msg in episode.turns if msg.role == Role.USER)
    assistant_text = " ".join(msg.content for msg in episode.turns if msg.role == Role.ASSISTANT)

    user_tokens = tokenize(user_text, model=metric.tokenizer_model)
    combined_tokens = tokenize(
        f"{user_text} {assistant_text}".strip(),
        model=metric.tokenizer_model,
    )

    assert value.metadata["token_ids"] == user_tokens
    assert value.metadata["token_count"] == len(user_tokens)
    assert value.metadata["token_ids"] != combined_tokens


def test_ttr_metric_empty_episode() -> None:
    """Test TTR with no user turns."""
    metric = TypeTokenRatioMetric()

    episode = _make_episode(
        episode_id="ep5",
        user_turns=[],
    )

    value = metric.evaluate(episode)

    assert value.metadata["token_count"] == 0
    assert len(value.metadata["token_ids"]) == 0


def test_ttr_metric_aggregate_single_episode() -> None:
    """Test aggregation with a single episode."""
    metric = TypeTokenRatioMetric()

    episode = _make_episode(
        episode_id="ep1",
        user_turns=["the cat sat on the mat"],
    )

    value = metric.evaluate(episode)
    values = [value]
    aggregate = metric.aggregate(values)

    total_tokens, total_types = _corpus_counts(values)
    expected_ttr = 0.0 if total_tokens == 0 else total_types / total_tokens

    assert aggregate.metric_name == METRIC_NAME
    assert aggregate.sample_size == len(values)
    assert aggregate.standard_deviation == 0.0
    assert aggregate.extras["total_tokens"] == total_tokens
    assert aggregate.extras["total_types"] == total_types
    assert pytest.approx(aggregate.mean, rel=1e-3) == expected_ttr


def test_ttr_metric_aggregate_multiple_episodes() -> None:
    """Test true corpus-level TTR aggregation across multiple episodes."""
    metric = TypeTokenRatioMetric(min_tokens=2)

    episodes = [
        _make_episode(
            episode_id="ep1",
            user_turns=["hello world hello"],  # "hello" appears twice
        ),
        _make_episode(
            episode_id="ep2",
            user_turns=["hello again"],  # "hello" appears again in different episode
        ),
        _make_episode(
            episode_id="ep3",
            user_turns=["world"],  # "world" appears again
        ),
    ]

    values = [metric.evaluate(ep) for ep in episodes]
    aggregate = metric.aggregate(values)

    total_tokens, total_types = _corpus_counts(values)
    expected_ttr = 0.0 if total_tokens == 0 else total_types / total_tokens

    assert aggregate.sample_size == len(values)
    assert aggregate.extras["total_tokens"] == total_tokens
    assert aggregate.extras["total_types"] == total_types
    assert pytest.approx(aggregate.mean, rel=1e-3) == expected_ttr


def test_ttr_metric_aggregate_empty_values() -> None:
    """Test aggregation with no metric values."""
    metric = TypeTokenRatioMetric()
    aggregate = metric.aggregate([])

    assert aggregate.metric_name == METRIC_NAME
    assert aggregate.mean == 0.0
    assert aggregate.sample_size == 0
    assert aggregate.standard_deviation == 0.0
    assert aggregate.extras["total_tokens"] == 0
    assert aggregate.extras["total_types"] == 0


def test_ttr_metric_aggregate_zero_tokens() -> None:
    """Test aggregation when all episodes have zero tokens."""
    metric = TypeTokenRatioMetric()

    episodes = [
        _make_episode(episode_id="ep1", user_turns=[""]),
        _make_episode(episode_id="ep2", user_turns=[]),
    ]

    values = [metric.evaluate(ep) for ep in episodes]
    aggregate = metric.aggregate(values)

    assert aggregate.mean == 0.0
    assert aggregate.standard_deviation == 0.0
    assert aggregate.extras["total_tokens"] == 0
    assert aggregate.extras["total_types"] == 0


def test_ttr_metric_repeated_words() -> None:
    """Test TTR correctly handles repeated words with tiktoken."""
    metric = TypeTokenRatioMetric()

    episode = _make_episode(
        episode_id="ep1",
        user_turns=["cat cat cat dog dog"],
    )

    value = metric.evaluate(episode)
    token_ids = value.metadata["token_ids"]
    total_tokens, total_types = _corpus_counts([value])
    expected_ttr = 0.0 if total_tokens == 0 else total_types / total_tokens

    assert value.metadata["token_count"] == len(token_ids)
    assert total_tokens == len(token_ids)
    assert total_types == len(set(token_ids))

    aggregate = metric.aggregate([value])
    assert pytest.approx(aggregate.mean, rel=1e-3) == expected_ttr
    assert aggregate.standard_deviation == 0.0


def test_ttr_metric_custom_min_tokens() -> None:
    """Test custom min_tokens threshold."""
    metric = TypeTokenRatioMetric(min_tokens=10)

    episode = _make_episode(
        episode_id="ep1",
        user_turns=["one two three four five"],  # 5 tokens
    )

    value = metric.evaluate(episode)

    assert value.metadata["below_threshold"] is True
    assert value.metadata["min_tokens"] == 10  # noqa: PLR2004

    aggregate = metric.aggregate([value])
    assert aggregate.extras["min_tokens_threshold"] == 10  # noqa: PLR2004
    assert aggregate.extras["episodes_below_threshold"] == 1


def test_ttr_metric_custom_target_role() -> None:
    """Test analyzing ASSISTANT role instead of USER role."""
    metric = TypeTokenRatioMetric(target_role=Role.ASSISTANT)

    episode = _make_episode(
        episode_id="ep1",
        user_turns=["hello world"],
        assistant_turns=["hi there how can I help you today"],
    )

    value = metric.evaluate(episode)

    assistant_text = " ".join(msg.content for msg in episode.turns if msg.role == Role.ASSISTANT)
    expected_tokens = tokenize(assistant_text, model=metric.tokenizer_model)

    assert value.metadata["token_ids"] == expected_tokens
    assert value.metadata["token_count"] == len(expected_tokens)
    assert value.metadata["target_role"] == Role.ASSISTANT.value

    aggregate = metric.aggregate([value])
    assert aggregate.extras["target_role"] == Role.ASSISTANT.value


def test_ttr_metric_target_role_default_is_user() -> None:
    """Test that default target_role is USER."""
    metric = TypeTokenRatioMetric()

    episode = _make_episode(
        episode_id="ep1",
        user_turns=["user message"],
        assistant_turns=["assistant message that is longer"],
    )

    value = metric.evaluate(episode)

    user_text = " ".join(msg.content for msg in episode.turns if msg.role == Role.USER)
    expected_tokens = tokenize(user_text, model=metric.tokenizer_model)

    # Should analyze user messages by default
    assert value.metadata["token_ids"] == expected_tokens
    assert value.metadata["token_count"] == len(expected_tokens)
    assert value.metadata["target_role"] == Role.USER.value

    aggregate = metric.aggregate([value])
    assert aggregate.extras["target_role"] == Role.USER.value
