"""Utilities for serializing and persisting episode artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from mirrorbench.core.models.episodes import EpisodeArtifact, EpisodeSpec, ReferenceStats
from mirrorbench.core.models.messages import JudgeVerdict, Message, Role, TurnTelemetry
from mirrorbench.core.models.plan import EvalUnit
from mirrorbench.core.models.run import metric_value_to_dict
from mirrorbench.io.paths import Paths


def serialize_episode_artifact(artifact: EpisodeArtifact) -> dict[str, Any]:
    return {
        "spec": _serialize_episode_spec(artifact.spec),
        "turns": [_serialize_message(turn) for turn in artifact.turns],
        "telemetry": [_serialize_turn_telemetry(t) for t in artifact.telemetry],
        "judge_verdicts": [_serialize_judge_verdict(judge) for judge in artifact.judge_verdicts],
        "metric_values": {
            name: metric_value_to_dict(value) for name, value in artifact.metric_values.items()
        },
        "errors": list(artifact.errors),
    }


def dump_episode_artifact(
    *,
    paths: Paths,
    run_id: str,
    unit: EvalUnit,
    artifact: EpisodeArtifact,
) -> str:
    """Persist ``artifact`` to disk and return the relative path within the run directory."""

    run_dir = paths.run_dir(run_id)
    artifact_dir = run_dir / "artifacts" / unit.unit_id()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    file_path = artifact_dir / f"{artifact.spec.episode_id}.json"
    payload = serialize_episode_artifact(artifact)
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return str(file_path.relative_to(run_dir))


def _serialize_episode_spec(spec: EpisodeSpec) -> dict[str, Any]:
    return {
        "episode_id": spec.episode_id,
        "task_tag": spec.task_tag,
        "chat_history": [_serialize_message(message) for message in spec.chat_history],
        "references": spec.references,
        "metadata": spec.metadata,
        "reference_stats": _serialize_reference_stats(spec.reference_stats),
    }


def _serialize_message(message: Message) -> dict[str, Any]:
    return {
        "role": message.role.value if isinstance(message.role, Role) else str(message.role),
        "content": message.content,
        "message_id": message.message_id,
        "name": message.name,
        "timestamp": (
            message.timestamp.isoformat() if isinstance(message.timestamp, datetime) else None
        ),
        "metadata": dict(message.metadata),
    }


def _serialize_turn_telemetry(telemetry: TurnTelemetry | None) -> dict[str, Any] | None:
    if telemetry is None:
        return None

    return {
        "time_to_first_token": telemetry.time_to_first_token,
        "time_per_output_token": telemetry.time_per_output_token,
        "total_response_time": telemetry.total_response_time,
        "tokens_input": telemetry.tokens_input,
        "tokens_output": telemetry.tokens_output,
        "cost_usd": telemetry.cost_usd,
        "provider": telemetry.provider,
        "metadata": dict(telemetry.metadata),
    }


def _serialize_judge_verdict(verdict: JudgeVerdict) -> dict[str, Any]:
    return {
        "score": verdict.score,
        "label": verdict.label,
        "confidence": verdict.confidence,
        "reason": verdict.reason,
        "raw": dict(verdict.raw),
        "telemetry": _serialize_turn_telemetry(verdict.telemetry),
    }


def _serialize_reference_stats(stats: ReferenceStats | None) -> dict[str, Any] | None:
    if stats is None:
        return None

    return {
        "schema_version": stats.schema_version,
        "metrics": dict(stats.metrics),
        "distributions": dict(stats.distributions),
        "generated_at": stats.generated_at.isoformat() if stats.generated_at else None,
    }


__all__ = ["dump_episode_artifact", "serialize_episode_artifact"]
