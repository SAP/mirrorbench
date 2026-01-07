"""Tests for JSONLOASST1Mirror dataset."""

from __future__ import annotations

import json
from pathlib import Path

import mirrorbench.datasets  # noqa: F401 - ensure datasets register
from mirrorbench.core.constants import REGISTRY_GROUP_DATASETS
from mirrorbench.core.models.plan import DatasetSpec
from mirrorbench.core.registry import registry
from mirrorbench.io.paths import Paths

DATASET_NAME = "dataset:jsonl/oasst1_mirror"
REFERENCE_TURN_COUNT = 4


def _dataset(paths: Paths):
    factory = registry.factory(REGISTRY_GROUP_DATASETS, DATASET_NAME)
    return factory(paths=paths)


def test_oasst1_mirror_reads_records(tmp_path):
    """Test that the dataset correctly reads and parses OASST1 mirror records."""
    base = Path(tmp_path)
    jsonl_path = base / "oasst1_mirror.jsonl"
    rows = [
        {
            "dataset": "oasst1",
            "conversation_id": "oasst-uuid-1",
            "task_description": "The user is asking for technical assistance.",
            "turns": [
                {
                    "role": "user",
                    "content": "How do I install Python?",
                    "metadata": {"lang": "en", "rank": None, "message_id": "msg-1"},
                },
                {
                    "role": "assistant",
                    "content": "You can download Python from python.org",
                    "metadata": {"lang": "en", "rank": 0, "message_id": "msg-2"},
                },
                {
                    "role": "user",
                    "content": "Thanks! What about pip?",
                    "metadata": {"lang": "en", "rank": None, "message_id": "msg-3"},
                },
                {
                    "role": "assistant",
                    "content": "Pip is included with Python 3.4+",
                    "metadata": {"lang": "en", "rank": 0, "message_id": "msg-4"},
                },
            ],
            "metadata": {
                "language": "en",
                "tree_state": "ready_for_export",
                "review_result": True,
                "dataset_split": "train",
                "num_turns": 4,
                "num_user_turns": 2,
            },
        },
        {
            "dataset": "oasst1",
            "conversation_id": "oasst-uuid-2",
            "task_description": "The user wants to learn about ML.",
            "turns": [
                {"role": "user", "content": "What is machine learning?", "metadata": {}},
                {
                    "role": "assistant",
                    "content": "ML is a branch of AI.",
                    "metadata": {},
                },
            ],
            "metadata": {"language": "en", "num_turns": 2},
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
    assert first.episode_id == "oasst-uuid-1"
    assert first.task_tag == "chat"
    assert len(first.chat_history) == 4  # noqa: PLR2004
    assert first.chat_history[0].role.value == "user"
    assert first.chat_history[1].role.value == "assistant"
    assert first.chat_history[2].role.value == "user"
    assert first.chat_history[3].role.value == "assistant"
    assert first.metadata["task_description"] == "The user is asking for technical assistance."
    assert first.metadata["dataset"] == "oasst1"
    assert first.metadata["language"] == "en"
    assert first.metadata["tree_state"] == "ready_for_export"
    assert first.metadata["review_result"] is True
    assert first.metadata["num_turns"] == 4  # noqa: PLR2004
    assert first.metadata["num_user_turns"] == 2  # noqa: PLR2004
    assert "real_conversation" in first.references
    assert len(first.references["real_conversation"]) == REFERENCE_TURN_COUNT

    # Check second episode
    second = episodes[1]
    assert second.episode_id == "oasst-uuid-2"
    assert len(second.chat_history) == 2  # noqa: PLR2004
    assert "real_conversation" in second.references


def test_oasst1_mirror_with_limit(tmp_path):
    """Test that the dataset respects the limit parameter."""
    base = Path(tmp_path)
    jsonl_path = base / "oasst1_mirror.jsonl"
    rows = [
        {
            "dataset": "oasst1",
            "conversation_id": f"oasst-{i}",
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


def test_oasst1_mirror_multilingual_support(tmp_path):
    """Test that the dataset handles multiple languages."""
    base = Path(tmp_path)
    jsonl_path = base / "oasst1_mirror.jsonl"
    rows = [
        {
            "dataset": "oasst1",
            "conversation_id": "oasst-de-1",
            "task_description": "German conversation",
            "turns": [
                {
                    "role": "user",
                    "content": "Wie geht es dir?",
                    "metadata": {"lang": "de"},
                },
                {
                    "role": "assistant",
                    "content": "Mir geht es gut, danke!",
                    "metadata": {"lang": "de"},
                },
            ],
            "metadata": {"language": "de", "num_turns": 2},
        },
        {
            "dataset": "oasst1",
            "conversation_id": "oasst-en-1",
            "task_description": "English conversation",
            "turns": [
                {"role": "user", "content": "How are you?", "metadata": {"lang": "en"}},
                {
                    "role": "assistant",
                    "content": "I'm doing well!",
                    "metadata": {"lang": "en"},
                },
            ],
            "metadata": {"language": "en", "num_turns": 2},
        },
    ]
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")

    paths = Paths(base)
    dataset = _dataset(paths)
    spec = DatasetSpec(name=DATASET_NAME, split="default", params={"path": str(jsonl_path)})

    episodes = list(dataset.episodes(spec=spec, split="default"))
    assert len(episodes) == 2  # noqa: PLR2004
    assert episodes[0].metadata["language"] == "de"
    assert episodes[1].metadata["language"] == "en"


def test_oasst1_mirror_custom_task_tag(tmp_path):
    """Test that custom task tags are respected."""
    base = Path(tmp_path)
    jsonl_path = base / "oasst1_mirror.jsonl"
    rows = [
        {
            "dataset": "oasst1",
            "conversation_id": "oasst-qa",
            "task_tag": "qa",  # Custom task tag
            "task_description": "Q&A conversation",
            "turns": [
                {"role": "user", "content": "What is Python?", "metadata": {}},
                {
                    "role": "assistant",
                    "content": "Python is a programming language.",
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
    assert episodes[0].task_tag == "qa"
