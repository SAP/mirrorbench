"""Execution controller orchestrating manifests and run database interactions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog
from structlog import contextvars as structlog_contextvars

from mirrorbench.core.config import RunConfig, ScorecardConfig
from mirrorbench.core.constants import (
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
)
from mirrorbench.core.manifest import ManifestIO, RunManifest
from mirrorbench.core.models.plan import EvalUnit, PlanManifest
from mirrorbench.core.models.run import (
    EpisodeResult,
    MetricAggregate,
    RunMetadata,
    RunSummary,
    ScorecardResult,
    metric_value_from_dict,
    metric_value_to_dict,
    scorecard_result_from_dict,
    scorecard_result_to_dict,
)
from mirrorbench.core.run_db import RunDatabase, SQLiteRunDatabase
from mirrorbench.core.telemetry import get_meter, get_tracer
from mirrorbench.io.paths import Paths


class RunController:
    """Coordinate manifest persistence, run database updates, and telemetry."""

    def __init__(  # noqa: PLR0913
        self,
        run_id: str,
        plan_manifest: PlanManifest,
        run_config: RunConfig,
        paths: Paths | None = None,
        db: RunDatabase | None = None,
        manifest_io: ManifestIO | None = None,
        scorecards: list[ScorecardConfig] | None = None,
        *,
        revision: int = 0,
        previous_run_manifest: RunManifest | None = None,
    ) -> None:
        self.run_id = run_id
        self.plan_manifest = plan_manifest
        self.run_config = run_config
        self.paths = paths or Paths.default()
        self.manifest_io = manifest_io or ManifestIO(self.paths)
        self.db = db or SQLiteRunDatabase(self.paths.run_db_path(run_id), run_id)
        self.db.initialize()
        self.revision = revision
        self.previous_run_manifest = previous_run_manifest
        history: list[dict[str, Any]] = []
        if previous_run_manifest is not None:
            prior = previous_run_manifest.extras.get("history")
            if isinstance(prior, list):
                history = [dict(entry) for entry in prior]
        self._history = history
        self.manifest_io.write_plan(run_id, plan_manifest, revision=revision)
        self._logger = structlog.get_logger(__name__).bind(run_id=run_id)
        structlog_contextvars.bind_contextvars(run_id=run_id, revision=revision)
        self._unit_map: dict[str, EvalUnit] = {}
        self._telemetry_totals: dict[str, float] = {}
        self.scorecards = scorecards or []
        self._aggregates: list[MetricAggregate] = []
        self._executed_units: list[str] = []
        self._tracer = get_tracer("mirrorbench.run")
        self._meter = get_meter("mirrorbench.run")
        self._run_span_cm: Any = None
        self._run_span: Any = None
        self._unit_spans: dict[str, tuple[Any, Any]] = {}
        self._counter_completed = self._meter.create_counter(
            "mirrorbench.units.completed", unit="1", description="Completed evaluation units"
        )
        self._counter_failed = self._meter.create_counter(
            "mirrorbench.units.failed", unit="1", description="Failed evaluation units"
        )
        self._counter_cancelled = self._meter.create_counter(
            "mirrorbench.units.cancelled", unit="1", description="Cancelled evaluation units"
        )
        self._counter_retries = self._meter.create_counter(
            "mirrorbench.units.retries", unit="1", description="Retries attempted per unit"
        )

    # ------------------------------------------------------------------
    # Run-level lifecycle
    # ------------------------------------------------------------------
    def on_run_start(self) -> None:
        existing = self.db.load_run_metadata()
        created_at = existing.created_at if existing is not None else datetime.now(UTC)
        extra = dict(existing.extra) if existing is not None else {}
        extra["status"] = STATUS_RUNNING
        extra["revision"] = self.revision
        metadata = RunMetadata(
            run_id=self.run_id,
            planner_version=self.plan_manifest.planner_version,
            engine=self.run_config.engine,
            created_at=created_at,
            extra=extra,
        )
        self.db.record_run(metadata)
        self._run_span_cm = self._tracer.start_as_current_span(
            "mirrorbench.run",
            attributes={
                "mirrorbench.run_id": self.run_id,
                "mirrorbench.engine": self.run_config.engine,
                "mirrorbench.revision": self.revision,
                "mirrorbench.planner_version": self.plan_manifest.planner_version,
            },
        )
        self._run_span = self._run_span_cm.__enter__()
        self._logger.info(
            "run_start",
            revision=self.revision,
            resume=existing is not None,
        )

    def on_run_end(self, status: str = STATUS_COMPLETED) -> RunSummary:
        summary = self.db.load_run_summary()
        if not summary.aggregates and self._aggregates:
            summary.aggregates.extend(self._aggregates)

        if self._telemetry_totals:
            summary.telemetry_stats.update(self._telemetry_totals)

        scorecards = self._compute_scorecards(summary.aggregates)
        summary.scorecards = scorecards
        summary_dict = _serialize_summary(summary)

        existing_summary: dict[str, Any] | None = None
        summary_path = self.paths.summary_path(self.run_id)
        if summary_path.exists():
            try:
                existing_summary = self.paths.load_run_summary(self.run_id)
            except FileNotFoundError:  # pragma: no cover - race guard
                existing_summary = None

        if existing_summary:
            merged_dict = _merge_summary_dicts(existing_summary, summary_dict)
            summary.units = [EvalUnit.model_validate(item) for item in merged_dict["units"]]
            summary.aggregates = [
                MetricAggregate(
                    metric_name=item["metric_name"],
                    mean=item.get("mean", 0.0),
                    standard_deviation=item.get("standard_deviation"),
                    confidence_interval=item.get("confidence_interval"),
                    p_value=item.get("p_value"),
                    sample_size=item.get("sample_size", 0),
                    extras=item.get("extras") or {},
                )
                for item in merged_dict["aggregates"]
            ]
            summary.episode_results = [
                EpisodeResult(
                    unit=EvalUnit.model_validate(entry["unit"]),
                    episode_id=entry["episode_id"],
                    status=entry["status"],
                    metric_values={
                        name: metric_value_from_dict(payload)
                        for name, payload in (entry.get("metric_values") or {}).items()
                    },
                    telemetry_summary=entry.get("telemetry_summary", {}),
                    artifact_path=entry.get("artifact_path"),
                    summary=entry.get("summary"),
                    duration_s=entry.get("duration_s"),
                )
                for entry in merged_dict["episode_results"]
            ]
            summary.telemetry_stats = merged_dict.get("telemetry_stats", {})
            summary.notes = list(merged_dict.get("notes", []))
            summary.scorecards = [
                scorecard_result_from_dict(item) for item in merged_dict.get("scorecards", [])
            ]
            summary_dict = merged_dict

        # Count episodes by status for easy visibility
        episodes_successful = sum(
            1 for ep in summary.episode_results if ep.status == STATUS_COMPLETED
        )
        episodes_failed = sum(1 for ep in summary.episode_results if ep.status == STATUS_FAILED)
        episodes_total = len(summary.episode_results)

        # Count units by status (from database)
        units_completed = sum(1 for unit_id in self._executed_units)
        units_total = len(self.plan_manifest.units)

        updated_extra = dict(summary.run.extra)
        updated_extra.update(
            {
                "status": status,
                "summary_json": summary_dict,
                "notes": summary.notes,
                "revision": self.revision,
                "scorecard_results": [
                    scorecard_result_to_dict(result) for result in summary.scorecards
                ],
                "episodes_successful": episodes_successful,
                "episodes_failed": episodes_failed,
                "episodes_total": episodes_total,
                "units_completed": units_completed,
                "units_total": units_total,
            }
        )
        summary.run.extra = updated_extra
        self.db.record_run(
            RunMetadata(
                run_id=self.run_id,
                planner_version=self.plan_manifest.planner_version,
                engine=self.run_config.engine,
                created_at=summary.run.created_at,
                extra=updated_extra,
            )
        )
        for scorecard in scorecards:
            self.db.record_scorecard(scorecard)

        self.paths.save_run_summary(self.run_id, summary_dict)

        timestamp = datetime.now(UTC).isoformat()
        self._history.append(
            {
                "revision": self.revision,
                "timestamp": timestamp,
                "planned_units": len(self.plan_manifest.units),
                "executed_units": len(self._executed_units),
            }
        )

        run_manifest = RunManifest(
            plan=self.plan_manifest,
            run_config=self.run_config,
            scorecards=self.scorecards,
            extras={
                "summary_path": str(self.paths.summary_path(self.run_id)),
                "status": status,
                "revision": self.revision,
                "executed_units": list(self._executed_units),
                "history": list(self._history),
            },
        )
        self.manifest_io.write_run(self.run_id, run_manifest)
        if self._run_span is not None and self._run_span_cm is not None:
            self._run_span.set_attribute("mirrorbench.status", status)
            self._run_span_cm.__exit__(None, None, None)
            self._run_span = None
            self._run_span_cm = None
        self._logger.info(
            "run_end",
            revision=self.revision,
            executed_units=len(self._executed_units),
            episodes_successful=episodes_successful,
            episodes_failed=episodes_failed,
            episodes_total=episodes_total,
        )
        return summary

    # ------------------------------------------------------------------
    # Unit lifecycle
    # ------------------------------------------------------------------
    def on_unit_start(self, unit: EvalUnit) -> str:
        unit_id = unit.unit_id()
        self.db.record_unit(unit, status=STATUS_RUNNING)
        self._unit_map[unit_id] = unit
        structlog_contextvars.bind_contextvars(unit_id=unit_id)
        span_cm = self._tracer.start_as_current_span(
            "mirrorbench.unit",
            attributes={
                "mirrorbench.run_id": self.run_id,
                "mirrorbench.unit_id": unit_id,
                "mirrorbench.proxy": unit.proxy_name,
                "mirrorbench.dataset": unit.dataset_name,
                "mirrorbench.metric": unit.metric_name,
                "mirrorbench.seed": unit.seed,
            },
        )
        span = span_cm.__enter__()
        self._unit_spans[unit_id] = (span_cm, span)
        self._logger.info(
            "unit_start",
            unit_id=unit_id,
            proxy=unit.proxy_name,
            dataset=unit.dataset_name,
            metric=unit.metric_name,
        )
        return unit_id

    def on_unit_end(self, unit: EvalUnit, status: str = STATUS_COMPLETED) -> None:
        unit_id = unit.unit_id()
        self.db.record_unit(unit, status=status)
        span_entry = self._unit_spans.pop(unit_id, None)
        if span_entry is not None:
            span_cm, span = span_entry
            span.set_attribute("mirrorbench.status", status)
            span_cm.__exit__(None, None, None)
        if status == STATUS_COMPLETED:
            self._counter_completed.add(1)
        elif status == STATUS_FAILED:
            self._counter_failed.add(1)
        elif status == STATUS_CANCELLED:
            self._counter_cancelled.add(1)
        structlog_contextvars.unbind_contextvars("unit_id")
        self._logger.info("unit_end", unit_id=unit_id, status=status)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def on_episode_start(self, unit: EvalUnit, episode_id: str) -> None:
        self._logger.debug("episode_start", unit_id=unit.unit_id(), episode_id=episode_id)

    def on_episode_end(self, unit: EvalUnit, episode_result: EpisodeResult) -> None:
        unit_id = unit.unit_id()
        self.db.record_episode_result(
            unit_id,
            episode_result,
            status=episode_result.status,
            duration_s=episode_result.duration_s,
            summary=episode_result.summary,
        )
        self._logger.debug("episode_end", unit_id=unit_id, episode_id=episode_result.episode_id)

    # ------------------------------------------------------------------
    # Metrics and telemetry
    # ------------------------------------------------------------------
    def on_metric_aggregate(self, unit: EvalUnit, aggregate: MetricAggregate) -> None:
        unit_id = unit.unit_id()
        self._logger.debug("record_metric", unit_id=unit_id, mean=aggregate.mean)
        self._aggregates.append(aggregate)
        self.db.record_metric_aggregate(unit_id, aggregate)
        self._logger.debug("metric_aggregate", unit_id=unit_id, metric=aggregate.metric_name)

    def on_unit_telemetry(self, unit: EvalUnit, telemetry: dict[str, float]) -> None:
        unit_id = unit.unit_id()
        self.db.record_unit_telemetry(unit_id, telemetry)
        for key, value in telemetry.items():
            if isinstance(value, int | float):
                self._telemetry_totals[key] = self._telemetry_totals.get(key, 0.0) + float(value)
        retries = telemetry.get("retries")
        if isinstance(retries, int | float) and retries:
            self._counter_retries.add(int(retries))
        self._logger.debug("unit_telemetry", unit_id=unit_id, telemetry=telemetry)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def close(self) -> None:
        self.db.close()
        structlog_contextvars.unbind_contextvars("run_id", "revision")

    def register_executed_units(self, unit_ids: list[str]) -> None:
        """Record executed unit ids for manifest revision bookkeeping."""

        self._executed_units = list(unit_ids)

    def _compute_scorecards(self, aggregates: list[MetricAggregate]) -> list[ScorecardResult]:
        lookup = {aggregate.metric_name: aggregate for aggregate in aggregates}
        results: list[ScorecardResult] = []
        for config in self.scorecards:
            missing = {metric for metric in config.weights if metric not in lookup}
            if missing:
                self._logger.warning(
                    "scorecard_missing_metrics",
                    scorecard=config.name,
                    missing=sorted(missing),
                )
                results.append(
                    ScorecardResult(
                        name=config.name,
                        score=None,
                        weights=dict(config.weights),
                        missing_metrics=missing,
                        extras={"components": {}},
                    )
                )
                continue

            components: dict[str, float] = {}
            score = 0.0
            for metric_name, weight in config.weights.items():
                aggregate = lookup[metric_name]
                components[metric_name] = aggregate.mean
                score += aggregate.mean * weight

            results.append(
                ScorecardResult(
                    name=config.name,
                    score=score,
                    weights=dict(config.weights),
                    extras={"components": components},
                )
            )

        return results


def _serialize_summary(summary: RunSummary) -> dict[str, Any]:
    return {
        "run": summary.run.model_dump(mode="json"),
        "units": [unit.model_dump(mode="json") for unit in summary.units],
        "aggregates": [
            {
                "metric_name": agg.metric_name,
                "mean": agg.mean,
                "standard_deviation": agg.standard_deviation,
                "confidence_interval": agg.confidence_interval,
                "p_value": agg.p_value,
                "sample_size": agg.sample_size,
                "extras": agg.extras,
            }
            for agg in summary.aggregates
        ],
        "episode_results": [
            {
                "unit": episode.unit.model_dump(mode="json"),
                "episode_id": episode.episode_id,
                "status": episode.status,
                "metric_values": {
                    name: metric_value_to_dict(value)
                    for name, value in episode.metric_values.items()
                },
                "telemetry_summary": episode.telemetry_summary,
                "artifact_path": episode.artifact_path,
                "summary": episode.summary,
                "duration_s": episode.duration_s,
            }
            for episode in summary.episode_results
        ],
        "telemetry_stats": summary.telemetry_stats,
        "notes": summary.notes,
        "scorecards": [scorecard_result_to_dict(card) for card in summary.scorecards],
    }


def _unit_key_from_dict(unit_dict: dict[str, Any]) -> str:
    return "|".join(
        [
            str(unit_dict.get("proxy_name", "")),
            str(unit_dict.get("dataset_name", "")),
            str(unit_dict.get("metric_name", "")),
            str(unit_dict.get("seed", 0)),
        ]
    )


def _merge_units(
    previous: list[dict[str, Any]], current: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in previous:
        merged[_unit_key_from_dict(item)] = item
    for item in current:
        merged[_unit_key_from_dict(item)] = item
    return [merged[key] for key in sorted(merged)]


def _merge_aggregates(
    previous: list[dict[str, Any]], current: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {
        item.get("metric_name", ""): item for item in previous if item.get("metric_name")
    }
    for item in current:
        name = item.get("metric_name")
        if not name:
            continue
        merged[name] = item
    return [merged[name] for name in sorted(merged)]


def _merge_episode_results(
    previous: list[dict[str, Any]], current: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for item in previous:
        key = (_unit_key_from_dict(item.get("unit", {})), str(item.get("episode_id")))
        merged[key] = item
    for item in current:
        key = (_unit_key_from_dict(item.get("unit", {})), str(item.get("episode_id")))
        merged[key] = item
    return [merged[key] for key in sorted(merged)]


def _merge_telemetry(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    merged = dict(previous)
    for key, value in current.items():
        if (
            key in merged
            and isinstance(merged[key], float | int)
            and isinstance(value, float | int)
        ):
            merged[key] = float(merged[key]) + float(value)
        else:
            merged[key] = value
    return merged


def _merge_scorecards(
    previous: list[dict[str, Any]], current: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in previous:
        name = str(item.get("name"))
        merged[name] = item
    for item in current:
        name = str(item.get("name"))
        merged[name] = item
    return [merged[name] for name in sorted(merged)]


def _merge_summary_dicts(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    merged_run = dict(current.get("run", {}))
    prev_extra = dict(previous.get("run", {}).get("extra", {}))
    curr_extra = dict(merged_run.get("extra", {}))
    prev_extra.pop("summary_json", None)
    curr_extra.pop("summary_json", None)
    merged_run["extra"] = {**prev_extra, **curr_extra}

    merged = {
        "run": merged_run,
        "units": _merge_units(previous.get("units", []), current.get("units", [])),
        "aggregates": _merge_aggregates(
            previous.get("aggregates", []), current.get("aggregates", [])
        ),
        "episode_results": _merge_episode_results(
            previous.get("episode_results", []), current.get("episode_results", [])
        ),
        "telemetry_stats": _merge_telemetry(
            previous.get("telemetry_stats", {}), current.get("telemetry_stats", {})
        ),
        "notes": list(dict.fromkeys(previous.get("notes", []) + current.get("notes", []))),
        "scorecards": _merge_scorecards(
            previous.get("scorecards", []), current.get("scorecards", [])
        ),
    }
    return merged
