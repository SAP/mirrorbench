"""Tests for JSONLChatbotArenaMirror dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mirrorbench.datasets  # noqa: F401 - ensure datasets register
from mirrorbench.core.constants import REGISTRY_GROUP_DATASETS
from mirrorbench.core.models.plan import DatasetSpec
from mirrorbench.core.registry import registry
from mirrorbench.datasets.base_dataset import DatasetError
from mirrorbench.io.paths import Paths

DATASET_NAME = "dataset:jsonl/chatbot_arena_mirror"
REFERENCE_TURN_COUNT = 2


def _dataset(paths: Paths):
    factory = registry.factory(REGISTRY_GROUP_DATASETS, DATASET_NAME)
    return factory(paths=paths)


def test_chatbot_arena_mirror_reads_records(tmp_path):
    """Test that the dataset correctly reads and parses chatbot arena mirror records."""
    base = Path(tmp_path)
    jsonl_path = base / "chatbot_arena_mirror.jsonl"
    rows = [
        {
            "dataset": "chatbot_arena",
            "conversation_id": "test-conv-1",
            "task_description": "The user is asking for help with a programming task.",
            "turns": [
                {"role": "user", "content": "How do I sort a list in Python?", "metadata": {}},
                {
                    "role": "assistant",
                    "content": "You can use the sorted() function or the .sort() method.",
                    "metadata": {},
                },
            ],
            "metadata": {
                "question_id": "q123",
                "arena_winner": "model_a",
                "selected_model": "gpt-4",
                "judge": "human_judge",
                "language": "en",
                "conversation_length": 2,
            },
        },
        {
            "dataset": "chatbot_arena",
            "conversation_id": "test-conv-2",
            "task_description": "The user wants to learn about machine learning.",
            "turns": [
                {
                    "role": "user",
                    "content": "What is machine learning?",
                    "metadata": {"topic": "ml"},
                },
                {
                    "role": "assistant",
                    "content": "Machine learning is a subset of AI.",
                    "metadata": {},
                },
            ],
            "metadata": {"language": "en", "conversation_length": 2},
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
    assert first.episode_id == "test-conv-1"
    assert first.task_tag == "chat_arena"
    assert len(first.chat_history) == 2  # noqa: PLR2004
    assert first.chat_history[0].role.value == "user"
    assert first.chat_history[1].role.value == "assistant"
    assert (
        first.metadata["task_description"] == "The user is asking for help with a programming task."
    )
    assert first.metadata["dataset"] == "chatbot_arena"
    assert first.metadata["arena_winner"] == "model_a"
    assert first.metadata["selected_model"] == "gpt-4"
    assert first.metadata["language"] == "en"
    assert "real_conversation" in first.references
    assert len(first.references["real_conversation"]) == REFERENCE_TURN_COUNT

    # Check second episode
    second = episodes[1]
    assert second.episode_id == "test-conv-2"
    assert len(second.chat_history) == 2  # noqa: PLR2004
    assert second.metadata["task_description"] == "The user wants to learn about machine learning."
    assert "real_conversation" in second.references
    assert len(second.references["real_conversation"]) == REFERENCE_TURN_COUNT


def test_chatbot_arena_mirror_with_limit(tmp_path):
    """Test that the dataset respects the limit parameter."""
    base = Path(tmp_path)
    jsonl_path = base / "chatbot_arena_mirror.jsonl"
    rows = [
        {
            "dataset": "chatbot_arena",
            "conversation_id": f"test-conv-{i}",
            "task_description": f"Task {i}",
            "turns": [
                {"role": "user", "content": f"Question {i}", "metadata": {}},
                {"role": "assistant", "content": f"Answer {i}", "metadata": {}},
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


def test_chatbot_arena_mirror_rejects_invalid_turns(tmp_path):
    """Test that the dataset raises errors for records with invalid turns."""
    base = Path(tmp_path)
    jsonl_path = base / "chatbot_arena_mirror.jsonl"

    # Test 1: Empty content should raise an error
    rows = [
        {
            "dataset": "chatbot_arena",
            "conversation_id": "test-conv-empty-content",
            "task_description": "Conversation with empty content",
            "turns": [
                {"role": "user", "content": "", "metadata": {}},
                {"role": "assistant", "content": "Response", "metadata": {}},
            ],
            "metadata": {},
        }
    ]
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")

    paths = Paths(base)
    dataset = _dataset(paths)
    spec = DatasetSpec(name=DATASET_NAME, split="default", params={"path": str(jsonl_path)})

    with pytest.raises(DatasetError, match="Missing content"):
        list(dataset.episodes(spec=spec, split="default"))

    # Test 2: Invalid role should raise an error
    rows = [
        {
            "dataset": "chatbot_arena",
            "conversation_id": "test-conv-invalid-role",
            "task_description": "Conversation with invalid role",
            "turns": [
                {"role": "invalid_role", "content": "Hello", "metadata": {}},
            ],
            "metadata": {},
        }
    ]
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")

    with pytest.raises(DatasetError, match="Invalid or missing role"):
        list(dataset.episodes(spec=spec, split="default"))

    # Test 3: Valid conversation should work fine
    rows = [
        {
            "dataset": "chatbot_arena",
            "conversation_id": "test-conv-valid",
            "task_description": "Valid conversation",
            "turns": [
                {"role": "user", "content": "Hello", "metadata": {}},
                {"role": "assistant", "content": "Hi there", "metadata": {}},
            ],
            "metadata": {},
        }
    ]
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")

    episodes = list(dataset.episodes(spec=spec, split="default"))
    assert len(episodes) == 1
    assert len(episodes[0].chat_history) == 2  # noqa: PLR2004
