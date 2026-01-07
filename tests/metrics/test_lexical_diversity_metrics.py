"""Unit tests for MATTR, HD-D, and Yule's K metrics."""

from __future__ import annotations

from statistics import mean, stdev
from typing import cast

import pytest

from mirrorbench.core.models.episodes import EpisodeArtifact, EpisodeSpec
from mirrorbench.core.models.messages import Message, Role
from mirrorbench.metrics.lexical.hdd import HDDMetric
from mirrorbench.metrics.lexical.mattr import MATTRMetric
from mirrorbench.metrics.lexical.yules_k import YulesKMetric
from mirrorbench.metrics.util.zscore import safe_z_score


def _make_episode(
    *,
    episode_id: str,
    human_user: str,
    proxy_user: str,
) -> EpisodeArtifact:
    """Construct an episode with aligned human and proxy user turns."""

    spec = EpisodeSpec(
        episode_id=episode_id,
        task_tag="test_task",
        chat_history=[Message(role=Role.USER, content=human_user)],
        references={
            "real_conversation": [
                {
                    "role": Role.USER.value,
                    "content": human_user,
                }
            ]
        },
    )

    turns = [
        Message(role=Role.USER, content=proxy_user),
    ]

    return EpisodeArtifact(spec=spec, turns=turns)


@pytest.mark.parametrize(
    "metric_factory",
    [
        lambda: MATTRMetric(window=3, tokenizer_model="gpt-4o", min_tokens=1),
        lambda: HDDMetric(sample_size=5, tokenizer_model="gpt-4o", min_tokens=1),
        lambda: YulesKMetric(tokenizer_model="gpt-4o", min_tokens=1),
    ],
)
def test_lexical_metrics_compute_z_scores(metric_factory) -> None:
    metric = metric_factory()

    episodes = [
        _make_episode(
            episode_id="ep1",
            human_user="hello world",
            proxy_user="hello hello world",
        ),
        _make_episode(
            episode_id="ep2",
            human_user="good morning",
            proxy_user="good good morning",
        ),
    ]

    values = [metric.evaluate(ep) for ep in episodes]

    proxy_scores = [
        (
            metric._score(cast(list[int], value.metadata["proxy_tokens"]))
            if value.metadata.get("proxy_tokens")
            else 0.0
        )
        for value in values
    ]
    human_scores = [
        metric._score(cast(list[int], value.metadata["human_tokens"]))
        for value in values
        if value.metadata.get("human_tokens")
    ]
    expected_mean = mean(human_scores) if human_scores else 0.0
    expected_std = stdev(human_scores) if len(human_scores) > 1 else 0.0
    expected_mean_proxy = mean(proxy_scores) if proxy_scores else 0.0

    aggregate = metric.aggregate(values)

    assert aggregate.metric_name == metric.info.name
    assert aggregate.sample_size == len(values)
    assert aggregate.extras["baseline_mean"] == pytest.approx(expected_mean)
    assert aggregate.extras["baseline_std"] == pytest.approx(expected_std)
    assert aggregate.extras["mean_proxy_raw"] == pytest.approx(expected_mean_proxy)
    if human_scores:
        assert aggregate.extras["mean_human_raw"] == pytest.approx(mean(human_scores))
    else:
        assert aggregate.extras["mean_human_raw"] == 0.0
    assert aggregate.extras.get("episodes_unstable_proxy", 0) == 0

    for value in values:
        proxy_raw = value.metadata["proxy_raw"]
        expected_z = safe_z_score(proxy_raw, expected_mean, expected_std)
        assert value.values == pytest.approx([expected_z])
        assert value.metadata["baseline_mean"] == pytest.approx(expected_mean)
        assert value.metadata["baseline_std"] == pytest.approx(expected_std)
        if value.metadata.get("human_tokens"):
            expected_human = metric._score(cast(list[int], value.metadata["human_tokens"]))
            assert value.metadata["human_raw"] == pytest.approx(expected_human)
        else:
            assert value.metadata["human_raw"] is None


def test_metric_handles_missing_human_reference() -> None:
    metric = MATTRMetric(window=2, tokenizer_model="gpt-4o", min_tokens=1)

    spec = EpisodeSpec(
        episode_id="missing",
        task_tag="test",
        chat_history=[],
    )
    episode = EpisodeArtifact(
        spec=spec,
        turns=[Message(role=Role.USER, content="synthetic conversation only")],
    )

    value = metric.evaluate(episode)
    assert value.metadata["human_tokens"] == []

    aggregate = metric.aggregate([value])

    assert aggregate.sample_size == 1
    assert aggregate.extras["baseline_sample_size"] == 0
    assert aggregate.extras["episodes_missing_human"] == 1
    assert value.metadata["human_raw"] is None
    assert value.values == [0.0]


def test_yules_k_basic_behaviour() -> None:
    metric = YulesKMetric(tokenizer_model="gpt-4o", min_tokens=1)

    unique_tokens_score = metric._score([1, 2, 3, 4])
    repeated_tokens_score = metric._score([1, 1, 1, 1])

    assert unique_tokens_score == 0.0
    assert repeated_tokens_score > unique_tokens_score
