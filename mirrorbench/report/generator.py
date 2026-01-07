"""Utilities for producing run reports."""

from __future__ import annotations

from typing import Any


def generate_json_report(summary: dict[str, Any]) -> dict[str, Any]:
    """Return a normalised JSON report derived from a persisted run summary."""

    run_block = summary.get("run", {})
    aggregates = summary.get("aggregates", [])
    units = summary.get("units", [])
    telemetry = summary.get("telemetry_stats", summary.get("telemetry", {}))
    notes = summary.get("notes", [])
    scorecards = summary.get("scorecards", [])

    return {
        "run": run_block,
        "aggregates": aggregates,
        "units": units,
        "telemetry": telemetry,
        "notes": notes,
        "scorecards": scorecards,
    }


__all__ = ["generate_json_report"]
