"""SQLite-backed run database implementation."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

from mirrorbench.core.constants import STATUS_CREATED, STATUS_PENDING
from mirrorbench.core.models.plan import EvalUnit
from mirrorbench.core.models.run import (
    EpisodeResult,
    MetricAggregate,
    RunMetadata,
    RunSummary,
    ScorecardResult,
    metric_value_from_dict,
    metric_value_to_dict,
    scorecard_result_from_dict,
)
from mirrorbench.core.run_db import schema
from mirrorbench.core.run_db.base import RunDatabase


def _isoformat(dt: datetime) -> str:
    return dt.isoformat()


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


class SQLiteRunDatabase(RunDatabase):
    """SQLite implementation of :class:`RunDatabase`."""

    def __init__(self, path: Path, run_id: str) -> None:
        super().__init__(path, run_id)
        self._conn: sqlite3.Connection | None = None
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            current_version = conn.execute("PRAGMA user_version").fetchone()[0]
            if current_version != schema.SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version = {schema.SCHEMA_VERSION}")
            for statement in schema.CREATE_TABLE_STATEMENTS:
                conn.executescript(statement)
            for statement in schema.CREATE_INDEXES:
                conn.execute(statement)
            conn.commit()
            self._conn = conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("RunDatabase has not been initialized")
        return self._conn

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def record_run(self, metadata: RunMetadata) -> None:
        with self._lock:
            conn = self._connection()
            status = metadata.extra.get("status", STATUS_CREATED)
            notes = metadata.extra.get("notes")
            summary_json = metadata.extra.get("summary_json")
            conn.execute(
                """
                INSERT INTO runs
                (run_id, created_at, status, engine, planner_version, summary_json, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    created_at=excluded.created_at,
                    status=excluded.status,
                    engine=excluded.engine,
                    planner_version=excluded.planner_version,
                    summary_json=excluded.summary_json,
                    notes=excluded.notes
                """,
                (
                    self.run_id,
                    _isoformat(metadata.created_at),
                    status,
                    metadata.engine,
                    metadata.planner_version,
                    json.dumps(summary_json) if summary_json is not None else None,
                    json.dumps(notes) if notes is not None else None,
                ),
            )
            conn.commit()

    def record_unit(self, eval_unit: EvalUnit, status: str = STATUS_PENDING) -> str:
        with self._lock:
            conn = self._connection()
            unit_id = eval_unit.unit_id()
            conn.execute(
                """
                INSERT INTO units
                (run_id, unit_id, user_proxy, dataset, dataset_split, metric, seed, judge, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, unit_id) DO UPDATE SET
                    user_proxy=excluded.user_proxy,
                    dataset=excluded.dataset,
                    dataset_split=excluded.dataset_split,
                    metric=excluded.metric,
                    seed=excluded.seed,
                    judge=excluded.judge,
                    status=excluded.status
                """,
                (
                    self.run_id,
                    unit_id,
                    eval_unit.proxy_name,
                    eval_unit.dataset_name,
                    eval_unit.dataset_split,
                    eval_unit.metric_name,
                    eval_unit.seed,
                    eval_unit.judge_name,
                    status,
                ),
            )
            conn.commit()
            return unit_id

    def record_episode_result(
        self,
        unit_id: str,
        episode_result: EpisodeResult,
        status: str,
        duration_s: float | None = None,
        summary: str | None = None,
    ) -> None:
        with self._lock:
            conn = self._connection()
            conn.execute(
                """
                INSERT OR REPLACE INTO episodes
                (run_id, unit_id, episode_id, status, duration_s, artifact_path, summary,
                 metric_values, telemetry_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    unit_id,
                    episode_result.episode_id,
                    status,
                    duration_s if duration_s is not None else episode_result.duration_s,
                    episode_result.artifact_path,
                    summary if summary is not None else episode_result.summary,
                    (
                        json.dumps(
                            {
                                name: metric_value_to_dict(value)
                                for name, value in episode_result.metric_values.items()
                            }
                        )
                        if episode_result.metric_values
                        else None
                    ),
                    (
                        json.dumps(episode_result.telemetry_summary)
                        if episode_result.telemetry_summary
                        else None
                    ),
                ),
            )
            conn.commit()

    def record_metric_aggregate(self, unit_id: str, aggregate: MetricAggregate) -> None:
        with self._lock:
            conn = self._connection()
            conn.execute(
                """
                INSERT OR REPLACE INTO metrics
                (run_id, unit_id, metric, mean, standard_deviation, confidence_interval,
                 p_value, sample_size, extras)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    unit_id,
                    aggregate.metric_name,
                    aggregate.mean,
                    aggregate.standard_deviation,
                    aggregate.confidence_interval,
                    aggregate.p_value,
                    aggregate.sample_size,
                    json.dumps(aggregate.extras) if aggregate.extras else None,
                ),
            )
            conn.commit()

    def record_unit_telemetry(self, unit_id: str, telemetry: dict[str, float]) -> None:
        with self._lock:
            conn = self._connection()
            rows = [
                (
                    self.run_id,
                    unit_id,
                    key,
                    json.dumps(value),
                )
                for key, value in telemetry.items()
            ]
            conn.executemany(
                """
                INSERT OR REPLACE INTO telemetry (run_id, unit_id, key, value)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def record_scorecard(self, scorecard: ScorecardResult) -> None:
        with self._lock:
            conn = self._connection()
            conn.execute(
                """
                INSERT OR REPLACE INTO scorecards
                (run_id, name, score, weights, missing_metrics, extras)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    scorecard.name,
                    scorecard.score,
                    json.dumps(scorecard.weights, sort_keys=True),
                    (
                        json.dumps(sorted(scorecard.missing_metrics))
                        if scorecard.missing_metrics
                        else None
                    ),
                    json.dumps(scorecard.extras, sort_keys=True) if scorecard.extras else None,
                ),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def load_run_summary(self) -> RunSummary:
        with self._lock:
            conn = self._connection()
            run_row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (self.run_id,),
            ).fetchone()
        if run_row is None:
            raise FileNotFoundError(f"Run '{self.run_id}' not recorded in database")

        metadata_extra: dict[str, Any] = {}
        if run_row["status"]:
            metadata_extra["status"] = run_row["status"]
        if run_row["summary_json"]:
            metadata_extra["summary_json"] = json.loads(run_row["summary_json"])
        if run_row["notes"]:
            metadata_extra["notes"] = json.loads(run_row["notes"])

        metadata = RunMetadata(
            run_id=run_row["run_id"],
            planner_version=run_row["planner_version"],
            engine=run_row["engine"],
            created_at=_parse_datetime(run_row["created_at"]),
            extra=metadata_extra,
        )

        unit_rows = conn.execute(
            "SELECT * FROM units WHERE run_id = ?",
            (self.run_id,),
        ).fetchall()
        units: dict[str, EvalUnit] = {}
        unit_list = []
        for row in unit_rows:
            # Handle backward compatibility: dataset_split may not exist in older schemas
            try:
                dataset_split = row["dataset_split"]
            except (KeyError, IndexError):
                dataset_split = "default"

            unit = EvalUnit(
                proxy_name=row["user_proxy"],
                dataset_name=row["dataset"],
                dataset_split=dataset_split,
                metric_name=row["metric"],
                seed=row["seed"],
                judge_name=row["judge"],
            )
            uid = row["unit_id"]
            units[uid] = unit
            unit_list.append(unit)

        episode_rows = conn.execute(
            "SELECT * FROM episodes WHERE run_id = ?",
            (self.run_id,),
        ).fetchall()
        print(f"Loading {len(episode_rows)} episodes for run '{self.run_id}'")
        episode_results = []
        for row in episode_rows:
            unit_entry = units.get(row["unit_id"])
            if unit_entry is None:
                continue
            raw_metric_values = json.loads(row["metric_values"]) if row["metric_values"] else {}
            metric_values = {
                name: metric_value_from_dict(payload) for name, payload in raw_metric_values.items()
            }
            telemetry_summary = json.loads(row["telemetry_json"]) if row["telemetry_json"] else {}
            episode = EpisodeResult(
                unit=unit_entry,
                episode_id=row["episode_id"],
                status=row["status"],
                metric_values=metric_values,
                telemetry_summary=telemetry_summary,
                artifact_path=row["artifact_path"],
                summary=row["summary"],
                duration_s=row["duration_s"],
            )
            episode_results.append(episode)

        metric_rows = conn.execute(
            "SELECT * FROM metrics WHERE run_id = ?",
            (self.run_id,),
        ).fetchall()
        aggregates = []
        for row in metric_rows:
            extras = json.loads(row["extras"]) if row["extras"] else {}
            aggregate = MetricAggregate(
                metric_name=row["metric"],
                mean=row["mean"],
                standard_deviation=row["standard_deviation"],
                confidence_interval=row["confidence_interval"],
                p_value=row["p_value"],
                sample_size=row["sample_size"],
                extras=extras,
            )
            aggregates.append(aggregate)

        telemetry_rows = conn.execute(
            "SELECT key, value FROM telemetry WHERE run_id = ?",
            (self.run_id,),
        ).fetchall()
        telemetry_stats = {
            row["key"]: json.loads(row["value"]) if row["value"] else None for row in telemetry_rows
        }

        notes = metadata_extra.get("notes")
        note_list = notes if isinstance(notes, list) else []

        scorecards = self.load_scorecards()

        return RunSummary(
            run=metadata,
            units=unit_list,
            aggregates=aggregates,
            episode_results=episode_results,
            telemetry_stats=telemetry_stats,
            notes=note_list,
            scorecards=scorecards,
        )

    def iter_units(self) -> Iterable[str]:
        with self._lock:
            conn = self._connection()
            rows = conn.execute(
                "SELECT unit_id FROM units WHERE run_id = ?",
                (self.run_id,),
            ).fetchall()
            return [row["unit_id"] for row in rows]

    def get_unit_statuses(self) -> dict[str, str]:
        with self._lock:
            conn = self._connection()
            rows = conn.execute(
                "SELECT unit_id, status FROM units WHERE run_id = ?",
                (self.run_id,),
            ).fetchall()
            return {row["unit_id"]: row["status"] for row in rows}

    def load_run_metadata(self) -> RunMetadata | None:
        with self._lock:
            conn = self._connection()
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (self.run_id,),
            ).fetchone()
        if row is None:
            return None

        extra: dict[str, Any] = {}
        if row["status"]:
            extra["status"] = row["status"]
        if row["summary_json"]:
            extra["summary_json"] = json.loads(row["summary_json"])
        if row["notes"]:
            extra["notes"] = json.loads(row["notes"])

        return RunMetadata(
            run_id=row["run_id"],
            planner_version=row["planner_version"],
            engine=row["engine"],
            created_at=_parse_datetime(row["created_at"]),
            extra=extra,
        )

    def load_scorecards(self) -> list[ScorecardResult]:
        with self._lock:
            conn = self._connection()
            rows = conn.execute(
                "SELECT name, score, weights, missing_metrics, extras FROM scorecards WHERE run_id = ?",
                (self.run_id,),
            ).fetchall()

        results: list[ScorecardResult] = []
        for row in rows:
            payload = {
                "name": row["name"],
                "score": row["score"],
                "weights": json.loads(row["weights"]) if row["weights"] else {},
                "missing_metrics": (
                    json.loads(row["missing_metrics"]) if row["missing_metrics"] else []
                ),
                "extras": json.loads(row["extras"]) if row["extras"] else {},
            }
            results.append(scorecard_result_from_dict(payload))
        return results
