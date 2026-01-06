"""Tests for the mirrorbench.metrics.judge suite."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from mirrorbench.core.constants import REGISTRY_GROUP_MODEL_CLIENTS
from mirrorbench.core.models.episodes import EpisodeArtifact, EpisodeSpec
from mirrorbench.core.models.messages import Message, Role
from mirrorbench.core.models.registry import ModelClientInfo
from mirrorbench.core.models.run import MetricValue
from mirrorbench.core.registry import registry
from mirrorbench.core.registry.entries import RegistryEntry
from mirrorbench.metrics.judge.calibration import (
    CONTROL_METADATA_KEY,
    HH_LABEL,
    PP_LABEL,
    apply_linear_calibration,
    derive_anchors,
)
from mirrorbench.metrics.judge.critique_then_revise import CritiqueThenReviseMetric
from mirrorbench.metrics.judge.pairwise_indistinguishability import (
    METRIC_NAME as PI_NAME,
    PairwiseIndistinguishabilityMetric,
)
from mirrorbench.metrics.judge.rubric_and_reason import (
    METRIC_NAME as RNR_NAME,
    RubricAndReasonMetric,
)
from mirrorbench.model_clients.base import ChatClient, ChatResponse

CRITIQUE_THEN_REVISE_REF_MEAN = 0.0
CRITIQUE_THEN_REVISE_SCORE_LEN = 2
RUBRIC_AND_REASON_SCORE = 1.0


def _register_client(name: str, factory: type[ChatClient]) -> None:
    try:
        registry.get(REGISTRY_GROUP_MODEL_CLIENTS, name)
    except Exception:  # pragma: no cover - already registered
        registry.register(
            RegistryEntry(
                group=REGISTRY_GROUP_MODEL_CLIENTS,
                name=name,
                factory=factory,
                metadata=None,
            )
        )


class PairwiseJudgeStub(ChatClient):
    """Judge stub that always favours the proxy conversation."""

    info = ModelClientInfo(
        name="test:pairwise-judge",
        provider="unit-test",
        capabilities={"chat"},
    )

    def __init__(self, *, rationale: str = "proxy is realistic") -> None:
        self.rationale = rationale

    def invoke(self, *, messages: list[Message], **_: Any) -> ChatResponse:
        user_prompt = messages[-1].content
        # Determine whether proxy conversation is presented as A or B
        conversation_a, conversation_b = user_prompt.split("[Conversation B]", maxsplit=1)
        proxy_in_a = "ProxyUser" in conversation_a
        verdict = "A" if proxy_in_a else "B"
        payload = json.dumps({"verdict": verdict, "reasoning": self.rationale})
        reply = Message(role=Role.ASSISTANT, content=payload)
        return ChatResponse(message=reply, usage=None, raw={"verdict": verdict})


class RubricJudgeStub(ChatClient):
    """Judge stub returning a fixed realism score."""

    info = ModelClientInfo(
        name="test:rubric-judge",
        provider="unit-test",
        capabilities={"chat"},
    )

    def __init__(self, *, score: float = 1.0, reasoning: str = "sounds human") -> None:
        self.score = score
        self.reasoning = reasoning

    def invoke(self, *, messages: list[Message], **_: Any) -> ChatResponse:
        verdict = "YES" if self.score >= 0.5 else "NO"  # noqa: PLR2004
        payload = json.dumps({"verdict": verdict, "reasoning": self.reasoning})
        reply = Message(role=Role.ASSISTANT, content=payload)
        return ChatResponse(
            message=reply, usage=None, raw={"verdict": verdict, "score": self.score}
        )


class CritiqueJudgeStub(ChatClient):
    """Judge stub implementing the critique + verdict contract."""

    info = ModelClientInfo(
        name="test:ctr-judge",
        provider="unit-test",
        capabilities={"chat"},
    )

    def __init__(self, *, score: float = 0.4) -> None:
        self.score = score
        self.last_critique = "- unnatural phrasing"

    def invoke(self, *, messages: list[Message], **_: Any) -> ChatResponse:
        verdict = "YES" if self.score >= 0.5 else "NO"  # noqa: PLR2004
        user_prompt = messages[-1].content
        if "Return JSON" in user_prompt:
            payload = json.dumps(
                {
                    "verdict": verdict,
                    "explanation": "final verdict",
                    "critique": self.last_critique,
                }
            )
            reply = Message(role=Role.ASSISTANT, content=payload)
            return ChatResponse(
                message=reply, usage=None, raw={"verdict": verdict, "score": self.score}
            )

        # Critique phase
        reply = Message(role=Role.ASSISTANT, content=self.last_critique)
        return ChatResponse(message=reply, usage=None, raw={"critique": self.last_critique})


@pytest.fixture(autouse=True)
def _register_stubs() -> None:
    _register_client("test:pairwise-judge", PairwiseJudgeStub)
    _register_client("test:rubric-judge", RubricJudgeStub)
    _register_client("test:ctr-judge", CritiqueJudgeStub)


def _make_episode(
    episode_id: str,
    *,
    proxy_conversation: Sequence[tuple[str, str]],
    real_conversation: Sequence[tuple[str, str]] | None = None,
    control: str | None = None,
) -> EpisodeArtifact:
    turns = [
        Message(role=Role.USER if role == "user" else Role.ASSISTANT, content=text)
        for role, text in proxy_conversation
    ]

    references: dict[str, Any] = {}
    chat_history: list[Message] = []
    if real_conversation is not None:
        real = [
            Message(role=Role.USER if role == "user" else Role.ASSISTANT, content=text)
            for role, text in real_conversation
        ]
        references["real_conversation"] = real
    else:
        real = []

    spec = EpisodeSpec(
        episode_id=episode_id,
        task_tag="test-task",
        chat_history=chat_history if real_conversation is None else [],
        references=references,
        metadata={"judge_control": control} if control else {},
    )

    return EpisodeArtifact(spec=spec, turns=turns)


def test_pairwise_metric_scores_proxy_win() -> None:
    metric = PairwiseIndistinguishabilityMetric(
        judge_client_name="test:pairwise-judge",
        num_judge_samples=2,
    )

    episode = _make_episode(
        "ep-pi",
        proxy_conversation=[("user", "ProxyUser: hello"), ("assistant", "hi")],
        real_conversation=[("user", "RealUser: hi"), ("assistant", "hello")],
    )

    value = metric.evaluate(episode)
    assert value.metric_name == PI_NAME
    assert value.values == [1.0]  # proxy always wins
    assert value.metadata["prompt_version"] == metric.prompt_version
    assert value.metadata["sample_count"] == CRITIQUE_THEN_REVISE_SCORE_LEN
    assert value.metadata["tie_count"] == 0
    assert len(episode.judge_verdicts) >= CRITIQUE_THEN_REVISE_SCORE_LEN

    aggregate = metric.aggregate([value])
    assert aggregate.mean == 1.0
    assert aggregate.extras["tie_rate"] == 0.0
    assert aggregate.extras["parity_gap"] == pytest.approx(0.5)  # 1.0 - 0.5
    assert aggregate.extras["calibration"]["enabled"] is True


def test_pairwise_metric_calibration() -> None:
    metric = PairwiseIndistinguishabilityMetric(judge_client_name="test:pairwise-judge")

    calibrated_values = [
        MetricValue(
            metric_name=PI_NAME,
            values=[0.6],
            metadata={
                "orders": [],
                "tie_count": 0,
                "sample_count": 1,
            },
        ),
        MetricValue(
            metric_name=PI_NAME,
            values=[0.9],
            metadata={
                CONTROL_METADATA_KEY: HH_LABEL,
                "orders": [],
                "tie_count": 0,
                "sample_count": 1,
            },
        ),
        MetricValue(
            metric_name=PI_NAME,
            values=[0.2],
            metadata={
                CONTROL_METADATA_KEY: PP_LABEL,
                "orders": [],
                "tie_count": 0,
                "sample_count": 1,
            },
        ),
    ]

    aggregate = metric.aggregate(calibrated_values)
    summary = aggregate.extras["calibration"]
    assert summary["enabled"] is True
    assert summary["hh_mean"] == pytest.approx(0.9)
    assert summary["pp_mean"] == pytest.approx(0.2)
    assert summary["calibrated_mean"] == pytest.approx(
        apply_linear_calibration([0.6], derive_anchors(calibrated_values))[0]
    )


def test_rubric_and_reason_metric() -> None:
    metric = RubricAndReasonMetric(judge_client_name="test:rubric-judge")

    episode = _make_episode(
        "ep-rnr",
        proxy_conversation=[("user", "Hello!"), ("assistant", "Hi there")],
        real_conversation=[("user", "Howdy!"), ("assistant", "Hello there")],
    )

    value = metric.evaluate(episode)
    assert value.metric_name == RNR_NAME
    assert value.values == [RUBRIC_AND_REASON_SCORE]
    assert value.metadata["reasoning"] == "sounds human"
    controls = value.metadata.get("controls")
    assert controls and "hh" in controls

    aggregate = metric.aggregate([value])
    assert aggregate.mean == RUBRIC_AND_REASON_SCORE
    assert aggregate.extras["prompt_version"] == metric.prompt_version
    assert aggregate.extras["reference_mean"] == pytest.approx(1.0)
    assert aggregate.extras["reference_count"] == 1


def test_critique_then_revise_metric() -> None:
    metric = CritiqueThenReviseMetric(
        judge_client_name="test:ctr-judge",
        num_judge_samples=2,
    )

    episode = _make_episode(
        "ep-ctr",
        proxy_conversation=[("user", "Hello!"), ("assistant", "Hi there")],
        real_conversation=[("user", "Hey!"), ("assistant", "Greetings")],
    )

    value = metric.evaluate(episode)
    assert value.values == [0.0]  # average of identical scores
    assert len(value.metadata["scores"]) == CRITIQUE_THEN_REVISE_SCORE_LEN
    assert value.metadata.get("controls")
    assert len(episode.judge_verdicts) >= CRITIQUE_THEN_REVISE_SCORE_LEN

    aggregate = metric.aggregate([value])
    assert aggregate.mean == CRITIQUE_THEN_REVISE_REF_MEAN
    assert aggregate.extras["prompt_version"] == metric.prompt_version
    assert aggregate.extras["reference_mean"] == pytest.approx(0.0)
    assert aggregate.extras["reference_count"] == 1


def test_derive_anchors_requires_both_controls() -> None:
    partial_values = [
        MetricValue(
            metric_name=PI_NAME,
            values=[0.7],
            metadata={
                CONTROL_METADATA_KEY: HH_LABEL,
                "orders": [],
                "tie_count": 0,
                "sample_count": 1,
            },
        )
    ]
    assert derive_anchors(partial_values) is None
