"""Tests for JSONLClariQMirror dataset."""

from __future__ import annotations

import json
from pathlib import Path

import mirrorbench.datasets  # noqa: F401 - ensure datasets register
from mirrorbench.core.constants import REGISTRY_GROUP_DATASETS
from mirrorbench.core.models.plan import DatasetSpec
from mirrorbench.core.registry import registry
from mirrorbench.io.paths import Paths

DATASET_NAME = "dataset:jsonl/clariq_mirror"
REFERENCE_TURN_COUNT = 3


def _dataset(paths: Paths):
    factory = registry.factory(REGISTRY_GROUP_DATASETS, DATASET_NAME)
    return factory(paths=paths)


def test_clariq_mirror_reads_records(tmp_path):
    """Test that the dataset correctly reads and parses ClariQ mirror records."""
    base = Path(tmp_path)
    jsonl_path = base / "clariq_mirror.jsonl"
    rows = [
        {
            "dataset": "clariq",
            "conversation_id": "clariq-1",
            "task_description": "The user is seeking information about a medical condition.",
            "turns": [
                {
                    "role": "user",
                    "content": "Tell me about diabetes",
                    "metadata": {"topic_id": "101", "facet_id": "F01"},
                },
                {
                    "role": "assistant",
                    "content": "Would you like to know about type 1 or type 2 diabetes?",
                    "metadata": {},
                },
                {"role": "user", "content": "Type 2 diabetes", "metadata": {}},
            ],
            "metadata": {
                "topic_id": "101",
                "facet_id": "F01",
                "facet": "Find information about type 2 diabetes.",
                "clarification_pairs": 1,
                "language": "en",
            },
        },
        {
            "dataset": "clariq",
            "conversation_id": "clariq-2",
            "task_description": "The user wants to learn about programming.",
            "turns": [
                {"role": "user", "content": "How do I learn Python?", "metadata": {}},
                {
                    "role": "assistant",
                    "content": "Do you have programming experience?",
                    "metadata": {},
                },
            ],
            "metadata": {"topic_id": "202", "language": "en"},
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
    assert first.episode_id == "clariq-1"
    assert first.task_tag == "clarification"
    assert len(first.chat_history) == 3  # noqa: PLR2004
    assert first.chat_history[0].role.value == "user"
    assert first.chat_history[1].role.value == "assistant"
    assert first.chat_history[2].role.value == "user"
    assert (
        first.metadata["task_description"]
        == "The user is seeking information about a medical condition."
    )
    assert first.metadata["dataset"] == "clariq"
    assert first.metadata["topic_id"] == "101"
    assert first.metadata["facet_id"] == "F01"
    assert first.metadata["facet"] == "Find information about type 2 diabetes."
    assert first.metadata["clarification_pairs"] == 1
    assert "real_conversation" in first.references
    assert len(first.references["real_conversation"]) == REFERENCE_TURN_COUNT

    # Check second episode
    second = episodes[1]
    assert second.episode_id == "clariq-2"
    assert len(second.chat_history) == 2  # noqa: PLR2004
    assert second.metadata["topic_id"] == "202"
    assert "real_conversation" in second.references


def test_clariq_mirror_with_limit(tmp_path):
    """Test that the dataset respects the limit parameter."""
    base = Path(tmp_path)
    jsonl_path = base / "clariq_mirror.jsonl"
    rows = [
        {
            "dataset": "clariq",
            "conversation_id": f"clariq-{i}",
            "task_description": f"Task {i}",
            "turns": [
                {"role": "user", "content": f"Question {i}", "metadata": {}},
                {"role": "assistant", "content": f"Clarification {i}", "metadata": {}},
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


def test_clariq_mirror_preserves_turn_metadata(tmp_path):
    """Test that turn-level metadata is preserved."""
    base = Path(tmp_path)
    jsonl_path = base / "clariq_mirror.jsonl"
    rows = [
        {
            "dataset": "clariq",
            "conversation_id": "clariq-meta",
            "task_description": "Test metadata preservation",
            "turns": [
                {
                    "role": "user",
                    "content": "Initial query",
                    "metadata": {"topic_id": "999", "facet_id": "F99"},
                },
                {
                    "role": "assistant",
                    "content": "Clarifying question",
                    "metadata": {"custom_field": "test_value"},
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
    # Check that turn metadata is preserved
    assert episode.chat_history[0].metadata["turn_index"] == 0
    assert episode.chat_history[1].metadata["turn_index"] == 1
    assert episode.chat_history[1].metadata["custom_field"] == "test_value"
