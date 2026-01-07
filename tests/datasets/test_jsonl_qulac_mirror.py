"""Tests for JSONLQulacMirror dataset."""

from __future__ import annotations

import json
from pathlib import Path

import mirrorbench.datasets  # noqa: F401 - ensure datasets register
from mirrorbench.core.constants import REGISTRY_GROUP_DATASETS
from mirrorbench.core.models.plan import DatasetSpec
from mirrorbench.core.registry import registry
from mirrorbench.io.paths import Paths

DATASET_NAME = "dataset:jsonl/qulac_mirror"


def _dataset(paths: Paths):
    factory = registry.factory(REGISTRY_GROUP_DATASETS, DATASET_NAME)
    return factory(paths=paths)


def test_qulac_mirror_reads_records(tmp_path):
    """Test that the dataset correctly reads and parses Qulac mirror records."""
    base = Path(tmp_path)
    jsonl_path = base / "qulac_mirror.jsonl"
    rows = [
        {
            "dataset": "qulac",
            "conversation_id": "qulac-1",
            "task_description": "The user is seeking information about a historical event.",
            "turns": [
                {
                    "role": "assistant",
                    "content": "Are you looking for a timeline or specific battles?",
                    "metadata": {"topic": "civil war battles"},
                },
                {
                    "role": "user",
                    "content": "I need a timeline",
                    "metadata": {},
                },
            ],
            "metadata": {
                "topic": "civil war battles",
                "topic_id": 100,
                "facet_id": 1,
                "facet_desc": "Find a timeline of civil war battles.",
                "topic_type": "faceted",
                "facet_type": "nav",
                "language": "en",
                "topic_desc": "What were the major battles in the US civil war?",
            },
        },
        {
            "dataset": "qulac",
            "conversation_id": "qulac-2",
            "task_description": "The user wants to learn about programming.",
            "turns": [
                {
                    "role": "assistant",
                    "content": "Do you want tutorials or documentation?",
                    "metadata": {},
                },
                {"role": "user", "content": "Tutorials please", "metadata": {}},
            ],
            "metadata": {"topic": "programming", "language": "en"},
        },
    ]
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")

    paths = Paths(base)
    dataset = _dataset(paths)
    spec = DatasetSpec(name=DATASET_NAME, split="default", params={"path": str(jsonl_path)})

    artifact = dataset.materialize(spec=spec, split="default")
    assert artifact.metadata["episode_count"] == 2  # noqa: PLR2004

    episodes = list(dataset.episodes(spec=spec, split="default"))
    assert len(episodes) == 2  # noqa: PLR2004

    # Check first episode
    first = episodes[0]
    assert first.episode_id == "qulac-1"
    assert first.task_tag == "query_clarification"
    assert len(first.chat_history) == 2  # noqa: PLR2004
    assert first.chat_history[0].role.value == "assistant"
    assert first.chat_history[1].role.value == "user"
    assert (
        first.metadata["task_description"]
        == "The user is seeking information about a historical event."
    )
    assert first.metadata["dataset"] == "qulac"
    assert first.metadata["topic"] == "civil war battles"
    assert first.metadata["topic_id"] == 100  # noqa: PLR2004
    assert first.metadata["facet_id"] == 1
    assert first.metadata["facet_desc"] == "Find a timeline of civil war battles."
    assert first.metadata["topic_type"] == "faceted"
    assert first.metadata["facet_type"] == "nav"
    assert first.metadata["topic_desc"] == "What were the major battles in the US civil war?"
    assert first.references == {}

    # Check second episode
    second = episodes[1]
    assert second.episode_id == "qulac-2"
    assert len(second.chat_history) == 2  # noqa: PLR2004
    assert second.metadata["topic"] == "programming"


def test_qulac_mirror_with_limit(tmp_path):
    """Test that the dataset respects the limit parameter."""
    base = Path(tmp_path)
    jsonl_path = base / "qulac_mirror.jsonl"
    rows = [
        {
            "dataset": "qulac",
            "conversation_id": f"qulac-{i}",
            "task_description": f"Task {i}",
            "turns": [
                {"role": "assistant", "content": f"Clarification {i}", "metadata": {}},
                {"role": "user", "content": f"Response {i}", "metadata": {}},
            ],
            "metadata": {"language": "en"},
        }
        for i in range(5)
    ]
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")

    paths = Paths(base)
    dataset = _dataset(paths)
    spec = DatasetSpec(name=DATASET_NAME, split="default", params={"path": str(jsonl_path)})

    episodes = list(dataset.episodes(spec=spec, split="default", limit=3))
    assert len(episodes) == 3  # noqa: PLR2004


def test_qulac_mirror_custom_task_tag(tmp_path):
    """Test that custom task tags are respected."""
    base = Path(tmp_path)
    jsonl_path = base / "qulac_mirror.jsonl"
    rows = [
        {
            "dataset": "qulac",
            "conversation_id": "qulac-custom",
            "task_tag": "clarification",  # Custom task tag
            "task_description": "Custom clarification",
            "turns": [
                {
                    "role": "assistant",
                    "content": "What do you mean?",
                    "metadata": {},
                },
                {"role": "user", "content": "I mean X", "metadata": {}},
            ],
            "metadata": {"language": "en"},
        }
    ]
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")

    paths = Paths(base)
    dataset = _dataset(paths)
    spec = DatasetSpec(name=DATASET_NAME, split="default", params={"path": str(jsonl_path)})

    episodes = list(dataset.episodes(spec=spec, split="default"))
    assert len(episodes) == 1
    assert episodes[0].task_tag == "clarification"


def test_qulac_mirror_assistant_first_turn(tmp_path):
    """Test that the dataset handles assistant-first conversations."""
    base = Path(tmp_path)
    jsonl_path = base / "qulac_mirror.jsonl"
    rows = [
        {
            "dataset": "qulac",
            "conversation_id": "qulac-assistant-first",
            "task_description": "Assistant initiates conversation",
            "turns": [
                {
                    "role": "assistant",
                    "content": "Would you like option A or B?",
                    "metadata": {},
                },
                {"role": "user", "content": "Option A", "metadata": {}},
                {
                    "role": "assistant",
                    "content": "Great choice!",
                    "metadata": {},
                },
            ],
            "metadata": {"language": "en"},
        }
    ]
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")

    paths = Paths(base)
    dataset = _dataset(paths)
    spec = DatasetSpec(name=DATASET_NAME, split="default", params={"path": str(jsonl_path)})

    episodes = list(dataset.episodes(spec=spec, split="default"))
    assert len(episodes) == 1

    episode = episodes[0]
    assert len(episode.chat_history) == 3  # noqa: PLR2004
    # First turn should be assistant
    assert episode.chat_history[0].role.value == "assistant"
    assert episode.chat_history[1].role.value == "user"
    assert episode.chat_history[2].role.value == "assistant"


def test_qulac_mirror_topic_metadata(tmp_path):
    """Test that topic-related metadata is preserved."""
    base = Path(tmp_path)
    jsonl_path = base / "qulac_mirror.jsonl"
    rows = [
        {
            "dataset": "qulac",
            "conversation_id": "qulac-topic-meta",
            "task_description": "Test topic metadata",
            "turns": [
                {
                    "role": "assistant",
                    "content": "Clarification",
                    "metadata": {"topic": "test_topic"},
                },
                {"role": "user", "content": "Response", "metadata": {}},
            ],
            "metadata": {
                "topic": "test_topic",
                "topic_id": 999,
                "facet_id": 99,
                "facet_desc": "Test facet",
                "topic_type": "ambiguous",
                "facet_type": "inf",
                "language": "en",
                "topic_desc": "Test topic description",
            },
        }
    ]
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")

    paths = Paths(base)
    dataset = _dataset(paths)
    spec = DatasetSpec(name=DATASET_NAME, split="default", params={"path": str(jsonl_path)})

    episodes = list(dataset.episodes(spec=spec, split="default"))
    assert len(episodes) == 1

    episode = episodes[0]
    # Verify all topic metadata is preserved
    assert episode.metadata["topic"] == "test_topic"
    assert episode.metadata["topic_id"] == 999  # noqa: PLR2004
    assert episode.metadata["facet_id"] == 99  # noqa: PLR2004
    assert episode.metadata["facet_desc"] == "Test facet"
    assert episode.metadata["topic_type"] == "ambiguous"
    assert episode.metadata["facet_type"] == "inf"
    assert episode.metadata["topic_desc"] == "Test topic description"
