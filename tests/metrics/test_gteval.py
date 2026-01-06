"""Unit tests for GTEvalMetric."""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest
from dotenv import load_dotenv

from mirrorbench.core.config import CacheConfig
from mirrorbench.core.constants import REGISTRY_GROUP_MODEL_CLIENTS
from mirrorbench.core.models.episodes import EpisodeArtifact, EpisodeSpec
from mirrorbench.core.models.errors import RegistryError
from mirrorbench.core.models.messages import Message, Role
from mirrorbench.core.models.registry import ModelClientInfo
from mirrorbench.core.registry import registry
from mirrorbench.core.registry.entries import RegistryEntry
from mirrorbench.io.paths import Paths
from mirrorbench.metrics.judge.gteval import GTEVAL_METRIC_INFO, GTEvalMetric
from mirrorbench.model_clients.base import ChatClient, ChatResponse
from mirrorbench.model_clients.caching_wrapper import CachingChatClient

METRIC_NAME = "metric:judge/gteval"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _has_azure_env() -> bool:
    return bool(os.getenv("AZURE_OPENAI_API_KEY")) and bool(os.getenv("AZURE_OPENAI_ENDPOINT"))


@dataclass
class FakeChatResponse:
    """Fake chat response for testing."""

    message: Message
    usage: dict[str, Any] | None = None


class FakeJudgeModelClient:
    """Fake model client that returns predetermined judge responses."""

    def __init__(self, *, score: float = 0.85, reasoning: str = "Test reasoning") -> None:
        self.score = score
        self.reasoning = reasoning
        self.last_messages: list[Message] = []

    def invoke(self, *, messages: list[Message], **_kwargs: Any) -> FakeChatResponse:
        """Fake invoke that returns a formatted judge response as JSON."""
        self.last_messages = messages
        response_json = json.dumps(
            {
                "reasoning": self.reasoning,
                "score": self.score,
            }
        )
        return FakeChatResponse(
            message=Message(role=Role.ASSISTANT, content=response_json),
        )


class CachedJudgeClient(ChatClient):
    """ChatClient-compatible stub to exercise caching behaviour."""

    info: ClassVar[ModelClientInfo] = ModelClientInfo(
        name="test:cached-judge",
        provider="unit-test",
        capabilities={"chat"},
    )

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, *, messages: list[Message], **_kwargs: Any) -> ChatResponse:
        self.calls += 1
        payload = json.dumps({"reasoning": "cached", "score": 0.5})
        reply = Message(role=Role.ASSISTANT, content=payload)
        usage = {"calls": self.calls}
        return ChatResponse(message=reply, usage=usage, raw={"calls": self.calls})

    def stream(self, *, messages: list[Message], **_kwargs: Any):  # pragma: no cover - unused
        yield from ()


def _make_episode(
    *,
    episode_id: str,
    real_conversation: list[tuple[str, str]],  # (role, content) pairs
    proxy_conversation: list[tuple[str, str]],  # (role, content) pairs
    store_real_in_references: bool = True,
) -> EpisodeArtifact:
    """Create a synthetic episode for testing GTEval.

    Args:
        episode_id: Episode identifier.
        real_conversation: List of (role, content) tuples for the real dataset conversation.
        proxy_conversation: List of (role, content) tuples for the proxy conversation.

    Returns:
        EpisodeArtifact with both conversations.
    """
    # Build real conversation (stored in spec.references)
    reference_conversation: list[Message] = []
    for role_str, content in real_conversation:
        role = Role.USER if role_str.lower() == "user" else Role.ASSISTANT
        reference_conversation.append(Message(role=role, content=content))

    references = {"real_conversation": reference_conversation} if store_real_in_references else {}
    chat_history = [] if store_real_in_references else list(reference_conversation)

    spec = EpisodeSpec(
        episode_id=episode_id,
        task_tag="test_task",
        chat_history=chat_history,
        references=references,
    )

    # Build proxy conversation (goes in artifact.turns)
    turns: list[Message] = []
    for role_str, content in proxy_conversation:
        role = Role.USER if role_str.lower() == "user" else Role.ASSISTANT
        turns.append(Message(role=role, content=content))

    return EpisodeArtifact(
        spec=spec,
        turns=turns,
    )


@pytest.fixture(autouse=True)
def _register_fake_judge_client() -> None:
    """Register a fake judge model client for testing."""
    # Only register if not already registered
    try:
        registry.get(REGISTRY_GROUP_MODEL_CLIENTS, "test:fake-judge")
    except RegistryError:
        # Not registered yet, register it
        entry = RegistryEntry(
            group=REGISTRY_GROUP_MODEL_CLIENTS,
            name="test:fake-judge",
            factory=FakeJudgeModelClient,
            metadata=None,
        )
        registry.register(entry)

    try:
        registry.get(REGISTRY_GROUP_MODEL_CLIENTS, "test:cached-judge")
    except RegistryError:
        registry.register(
            RegistryEntry(
                group=REGISTRY_GROUP_MODEL_CLIENTS,
                name="test:cached-judge",
                factory=CachedJudgeClient,
                metadata=None,
            )
        )


def test_gteval_metric_info() -> None:
    """Verify GTEval metric metadata is correctly defined."""
    assert GTEVAL_METRIC_INFO.name == METRIC_NAME
    assert GTEVAL_METRIC_INFO.needs_references is True
    assert GTEVAL_METRIC_INFO.needs_judge is True
    assert GTEVAL_METRIC_INFO.category == "human_likeness"


def test_gteval_with_model_client() -> None:
    """Test GTEval evaluation using internal judge with model client."""
    metric = GTEvalMetric(
        judge_client_name="test:fake-judge",
        judge_params={"score": 0.85, "reasoning": "High similarity observed"},
        rubric_version="1.0",
    )

    episode = _make_episode(
        episode_id="ep1",
        real_conversation=[
            ("user", "Hello there!"),
            ("assistant", "Hi! How can I help you?"),
            ("user", "What's the weather like?"),
        ],
        proxy_conversation=[
            ("user", "Hey!"),
            ("assistant", "Hello! What can I do for you?"),
            ("user", "Tell me about the weather."),
        ],
    )

    value = metric.evaluate(episode)

    assert value.metric_name == METRIC_NAME
    assert len(value.values) == 1
    assert value.values[0] == 0.85  # noqa: PLR2004
    assert value.metadata["rubric_version"] == "1.0"
    assert value.metadata["judge_client"] == "test:fake-judge"
    assert value.metadata["proxy_user_turn_count"] == 2  # noqa: PLR2004
    assert value.metadata["real_user_turn_count"] == 2  # noqa: PLR2004
    assert value.metadata["reasoning"] == "High similarity observed"
    assert len(episode.judge_verdicts) == 9  # noqa: PLR2004
    assert episode.judge_verdicts[0].score == 0.85  # noqa: PLR2004
    assert episode.judge_verdicts[0].raw == {
        "reasoning": "High similarity observed",
        "score": 0.85,
    }
    controls = value.metadata.get("controls")
    assert controls is not None
    assert pytest.approx(controls["hh"]["score"]) == 0.85  # noqa: PLR2004
    assert pytest.approx(controls["pp"]["score"]) == 0.85  # noqa: PLR2004
    assert episode.metric_values[METRIC_NAME].values == [0.85]
    assert episode.metric_values[METRIC_NAME].metric_name == METRIC_NAME


def test_gteval_missing_proxy_conversation_raises() -> None:
    """Test that GTEval raises error when proxy has no conversation."""
    metric = GTEvalMetric(judge_client_name="test:fake-judge")

    episode = _make_episode(
        episode_id="ep2",
        real_conversation=[("user", "hi")],
        proxy_conversation=[],  # Empty proxy conversation
    )

    with pytest.raises(ValueError, match="has no proxy conversation"):
        metric.evaluate(episode)


def test_gteval_missing_real_conversation_raises() -> None:
    """Test that GTEval raises error when real conversation is empty."""
    metric = GTEvalMetric(judge_client_name="test:fake-judge")

    episode = _make_episode(
        episode_id="ep3",
        real_conversation=[],  # Empty real conversation
        proxy_conversation=[("user", "hi")],
    )

    with pytest.raises(ValueError, match="has no real conversation"):
        metric.evaluate(episode)


def test_gteval_uses_cached_judge(tmp_path) -> None:
    metric = GTEvalMetric(judge_client_name="test:cached-judge")
    metric.configure_cache(
        cache_config=CacheConfig(enabled=True, ttl_seconds=None),
        paths=Paths(tmp_path / "mirrorbench"),
    )

    episode1 = _make_episode(
        episode_id="cached-1",
        real_conversation=[("user", "hi")],
        proxy_conversation=[("user", "hi")],
    )
    episode2 = _make_episode(
        episode_id="cached-2",
        real_conversation=[("user", "hi")],
        proxy_conversation=[("user", "hi")],
    )

    metric.evaluate(episode1)
    metric.evaluate(episode2)

    judge_client = metric._get_judge_client()
    assert isinstance(judge_client, CachingChatClient)
    # The wrapped client should have invoked the underlying delegate only once
    assert getattr(judge_client._delegate, "calls", 0) == 1  # type: ignore[attr-defined]


def test_gteval_aggregate_single_episode() -> None:
    """Test aggregation with a single episode."""
    metric = GTEvalMetric(judge_client_name="test:fake-judge")

    episode = _make_episode(
        episode_id="ep1",
        real_conversation=[("user", "hi"), ("assistant", "hello")],
        proxy_conversation=[("user", "hey"), ("assistant", "hi")],
    )

    value = metric.evaluate(episode)
    aggregate = metric.aggregate([value])

    assert aggregate.metric_name == METRIC_NAME
    assert aggregate.mean == 0.85  # noqa: PLR2004
    assert aggregate.standard_deviation == 0.0  # Only one episode
    assert aggregate.sample_size == 1
    assert aggregate.extras["total_proxy_user_turns"] == 1
    assert aggregate.extras["total_real_user_turns"] == 1


def test_gteval_aggregate_multiple_episodes() -> None:
    """Test aggregation across multiple episodes."""
    # Create metrics with different scores for each episode
    episodes = [
        _make_episode(
            episode_id="ep1",
            real_conversation=[("user", "a"), ("assistant", "b")],
            proxy_conversation=[("user", "c"), ("assistant", "d")],
        ),
        _make_episode(
            episode_id="ep2",
            real_conversation=[("user", "e"), ("assistant", "f")],
            proxy_conversation=[("user", "g"), ("assistant", "h")],
        ),
        _make_episode(
            episode_id="ep3",
            real_conversation=[("user", "i"), ("assistant", "j")],
            proxy_conversation=[("user", "k"), ("assistant", "l")],
        ),
    ]

    # Use different scores for variety
    scores = [0.8, 0.9, 0.7]
    values = []
    for episode, score in zip(episodes, scores, strict=False):
        metric = GTEvalMetric(
            judge_client_name="test:fake-judge",
            judge_params={"score": score},
        )
        evaluated = metric.evaluate(episode)
        assert evaluated.metric_name == METRIC_NAME
        values.append(evaluated)

    # Aggregate using any metric instance
    metric = GTEvalMetric(judge_client_name="test:fake-judge")
    aggregate = metric.aggregate(values)

    # Mean of [0.8, 0.9, 0.7] = 0.8
    assert pytest.approx(aggregate.mean, rel=1e-3) == 0.8  # noqa: PLR2004
    assert aggregate.sample_size == 3  # noqa: PLR2004
    assert aggregate.standard_deviation is not None
    assert aggregate.extras["count"] == 3  # noqa: PLR2004
    calibration = aggregate.extras["calibration"]
    assert calibration["enabled"] is True
    assert calibration["hh_mean"] is not None
    assert calibration["pp_mean"] is not None


def test_gteval_fallback_to_chat_history() -> None:
    """GTEval falls back to spec.chat_history when references are absent."""
    metric = GTEvalMetric(judge_client_name="test:fake-judge")

    episode = _make_episode(
        episode_id="ep_fallback",
        real_conversation=[
            ("user", "Hi"),
            ("assistant", "Hello"),
            ("user", "Thanks"),
        ],
        proxy_conversation=[
            ("user", "Hey"),
            ("assistant", "Hi"),
            ("user", "Cheers"),
        ],
        store_real_in_references=False,
    )

    # Ensure the episode spec mimics legacy datasets that only populate chat_history
    assert episode.spec.references == {}
    assert episode.spec.chat_history

    value = metric.evaluate(episode)

    assert len(value.values) == 1
    assert value.metadata["real_user_turn_count"] == 2  # noqa: PLR2004


def test_gteval_aggregate_empty_values() -> None:
    """Test aggregation with no metric values."""
    metric = GTEvalMetric(judge_client_name="test:fake-judge")
    aggregate = metric.aggregate([])

    assert aggregate.metric_name == METRIC_NAME
    assert aggregate.mean == 0.0
    assert aggregate.sample_size == 0
    assert aggregate.standard_deviation == 0.0


def test_gteval_parse_response() -> None:
    """Test parsing of judge response format."""
    metric = GTEvalMetric(judge_client_name="test:fake-judge")

    # Test successful JSON parsing
    json_response = json.dumps({"reasoning": "Good match", "score": 0.75})
    verdict = metric._parse_judge_response(json_response)
    assert verdict.score == 0.75  # noqa: PLR2004
    assert verdict.reason == "Good match"
    assert verdict.raw == {"reasoning": "Good match", "score": 0.75}

    # Test parsing with markdown code blocks
    json_with_markdown = "```json\n" + json_response + "\n```"
    verdict = metric._parse_judge_response(json_with_markdown)
    assert verdict.score == 0.75  # noqa: PLR2004
    assert verdict.reason == "Good match"

    # Test score validation (Pydantic will reject out-of-range scores)
    invalid_score_high = json.dumps({"reasoning": "Test", "score": 1.5})
    with pytest.raises(ValueError, match="does not match expected pydantic schema"):
        metric._parse_judge_response(invalid_score_high)

    invalid_score_low = json.dumps({"reasoning": "Test", "score": -0.5})
    with pytest.raises(ValueError, match="does not match expected pydantic schema"):
        metric._parse_judge_response(invalid_score_low)

    # Test missing score raises error
    missing_score = json.dumps({"reasoning": "No score provided"})
    with pytest.raises(ValueError, match="does not match expected pydantic schema"):
        metric._parse_judge_response(missing_score)

    # Test invalid JSON raises error
    with pytest.raises(ValueError, match="not valid JSON"):
        metric._parse_judge_response("This is not JSON")


@pytest.mark.real_azure_langchain
@pytest.mark.skipif(not _has_azure_env(), reason="Azure OpenAI credentials not configured")
def test_gteval_with_real_azure_judge_multi_episode() -> None:
    """Run GTEval against Azure OpenAI via LangChain for multiple episodes."""

    if importlib.util.find_spec("langchain_openai") is None:
        pytest.skip("langchain-openai package not installed")

    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    metric = GTEvalMetric(
        judge_client_name="client:langchain/chat",
        judge_params={
            "model_import": "langchain_openai.AzureChatOpenAI",
            "model_kwargs": {
                "azure_deployment": deployment,
                "api_version": api_version,
            },
        },
        rubric_version="azure-integration-test",
    )

    episodes = [
        _make_episode(
            episode_id="azure-ep1",
            real_conversation=[
                ("user", "Hi there!"),
                ("assistant", "Hello! How can I assist you today?"),
                ("user", "I'm looking for a dinner recipe with mushrooms."),
            ],
            proxy_conversation=[
                ("user", "Hey assistant!"),
                ("assistant", "Hi! What do you need help with today?"),
                ("user", "Need ideas for dinner that use mushrooms."),
            ],
        ),
        _make_episode(
            episode_id="azure-ep2",
            real_conversation=[
                ("user", "Can you help me plan a workout?"),
                ("assistant", "Sure! What equipment do you have available?"),
                ("user", "Just a yoga mat and resistance bands."),
                ("assistant", "Great, let's put something together."),
            ],
            proxy_conversation=[
                ("user", "I want a new exercise routine."),
                ("assistant", "Absolutely! Any preferences or equipment?"),
                ("user", "Only have a mat and some bands."),
                ("assistant", "Perfect, let's build a plan."),
            ],
        ),
    ]

    metric_values = []
    for episode in episodes:
        value = metric.evaluate(episode)
        assert value.metric_name == METRIC_NAME
        assert len(value.values) == 1
        assert 0.0 <= value.values[0] <= 1.0
        metric_values.append(value)
        assert metric.info.name in episode.metric_values
        assert episode.metric_values[metric.info.name] is value

    aggregate = metric.aggregate(metric_values)
    assert aggregate.sample_size == len(metric_values)
    assert 0.0 <= aggregate.mean <= 1.0
    assert aggregate.standard_deviation >= 0.0
    assert aggregate.extras["rubric_version"] == "azure-integration-test"


def test_gteval_multi_turn_conversations() -> None:
    """Test GTEval with realistic multi-turn conversations."""
    metric = GTEvalMetric(judge_client_name="test:fake-judge")

    episode = _make_episode(
        episode_id="ep7",
        real_conversation=[
            ("user", "I need help with my account"),
            ("assistant", "Sure, what seems to be the problem?"),
            ("user", "I can't log in"),
            ("assistant", "Let me help you reset your password."),
            ("user", "Thanks!"),
        ],
        proxy_conversation=[
            ("user", "Help me with my account please"),
            ("assistant", "Of course! What's the issue?"),
            ("user", "Login isn't working"),
            ("assistant", "I'll guide you through password reset."),
            ("user", "Thank you!"),
        ],
    )

    value = metric.evaluate(episode)

    assert value.metadata["proxy_user_turn_count"] == 3  # noqa: PLR2004
    assert value.metadata["real_user_turn_count"] == 3  # noqa: PLR2004
    assert value.values[0] == 0.85  # noqa: PLR2004


def test_gteval_custom_rubric_version() -> None:
    """Test GTEval with custom rubric version in metadata."""
    metric = GTEvalMetric(
        judge_client_name="test:fake-judge",
        rubric_version="3.0-beta",
    )

    episode = _make_episode(
        episode_id="ep8",
        real_conversation=[("user", "hi")],
        proxy_conversation=[("user", "hello")],
    )

    value = metric.evaluate(episode)
    assert value.metadata["rubric_version"] == "3.0-beta"

    aggregate = metric.aggregate([value])
    assert aggregate.extras["rubric_version"] == "3.0-beta"


def test_gteval_format_conversation() -> None:
    """Test conversation formatting for judge prompt."""
    metric = GTEvalMetric(judge_client_name="test:fake-judge")

    messages = [
        Message(role=Role.USER, content="Hello"),
        Message(role=Role.ASSISTANT, content="Hi there"),
        Message(role=Role.USER, content="How are you?"),
    ]

    formatted = metric._format_conversation(messages)

    expected = "USER: Hello\nASSISTANT: Hi there\nUSER: How are you?"
    assert formatted == expected


def test_gteval_judge_verdicts_have_telemetry() -> None:
    """Test that judge verdicts capture telemetry with component tag."""

    class TelemetryTrackingJudgeClient:
        """Fake judge client that returns telemetry with usage."""

        def invoke(self, *, messages: list[Message], **_kwargs: Any) -> ChatResponse:
            response_json = json.dumps({"reasoning": "Test with telemetry", "score": 0.75})
            usage = {
                "tokens_input": 100,
                "tokens_output": 50,
                "total_response_time": 1.5,
            }
            return ChatResponse(
                message=Message(role=Role.ASSISTANT, content=response_json),
                usage=usage,
                raw={},
            )

    # Register the telemetry tracking client
    try:
        registry.get(REGISTRY_GROUP_MODEL_CLIENTS, "test:telemetry-judge")
    except RegistryError:
        entry = RegistryEntry(
            group=REGISTRY_GROUP_MODEL_CLIENTS,
            name="test:telemetry-judge",
            factory=TelemetryTrackingJudgeClient,
            metadata=None,
        )
        registry.register(entry)

    metric = GTEvalMetric(judge_client_name="test:telemetry-judge", compute_controls=False)

    episode = _make_episode(
        episode_id="ep_telemetry",
        real_conversation=[("user", "hi"), ("assistant", "hello")],
        proxy_conversation=[("user", "hey"), ("assistant", "hi")],
    )

    metric.evaluate(episode)

    # Check that judge verdicts have telemetry
    assert len(episode.judge_verdicts) > 0
    for verdict in episode.judge_verdicts:
        assert verdict.telemetry is not None, "Judge verdict should have telemetry"
        assert verdict.telemetry.tokens_input == 100  # noqa: PLR2004
        assert verdict.telemetry.tokens_output == 50  # noqa: PLR2004
        assert verdict.telemetry.total_response_time == 1.5  # noqa: PLR2004
        # Verify component tag
        assert verdict.telemetry.metadata is not None
        assert "component" in verdict.telemetry.metadata
        assert verdict.telemetry.metadata["component"] == "judge"


def test_gteval_judge_telemetry_includes_rich_metadata() -> None:
    """Test that judge telemetry includes judge configuration details."""

    class RichTelemetryJudgeClient:
        """Fake judge client for testing rich telemetry."""

        def __init__(self, **kwargs: Any) -> None:
            # Accept any initialization parameters
            pass

        def invoke(self, *, messages: list[Message], **_kwargs: Any) -> ChatResponse:
            response_json = json.dumps({"reasoning": "Rich telemetry test", "score": 0.8})
            usage = {
                "tokens_input": 200,
                "tokens_output": 75,
                "total_response_time": 2.0,
            }
            return ChatResponse(
                message=Message(role=Role.ASSISTANT, content=response_json),
                usage=usage,
                raw={},
            )

    # Register the client
    try:
        registry.get(REGISTRY_GROUP_MODEL_CLIENTS, "test:rich-telemetry-judge")
    except RegistryError:
        entry = RegistryEntry(
            group=REGISTRY_GROUP_MODEL_CLIENTS,
            name="test:rich-telemetry-judge",
            factory=RichTelemetryJudgeClient,
            metadata=None,
        )
        registry.register(entry)

    metric = GTEvalMetric(
        judge_client_name="test:rich-telemetry-judge",
        judge_params={
            "model": "gpt-4o",
            "model_kwargs": {
                "azure_deployment": "gpt-4o-judge",
                "temperature": 0.0,
            },
        },
        compute_controls=False,
    )

    episode = _make_episode(
        episode_id="ep_rich_telemetry",
        real_conversation=[("user", "test")],
        proxy_conversation=[("user", "test")],
    )

    metric.evaluate(episode)

    # Check rich metadata in telemetry
    assert len(episode.judge_verdicts) > 0
    verdict = episode.judge_verdicts[0]
    assert verdict.telemetry is not None
    assert verdict.telemetry.metadata is not None

    # Verify judge configuration details
    assert verdict.telemetry.metadata["judge_client_name"] == "test:rich-telemetry-judge"
    assert verdict.telemetry.metadata["metric_name"] == METRIC_NAME

    # Verify model parameters
    assert verdict.telemetry.metadata["model"] == "gpt-4o"
    assert verdict.telemetry.metadata["azure_deployment"] == "gpt-4o-judge"
    assert verdict.telemetry.metadata["temperature"] == 0.0
