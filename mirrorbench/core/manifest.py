"""Utilities for persisting plan and run manifests to disk."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field

from mirrorbench.core.config import RunConfig, ScorecardConfig
from mirrorbench.core.models.plan import PlanManifest
from mirrorbench.io.paths import Paths


class RunManifest(BaseModel):
    """Persisted metadata describing an evaluation run."""

    schema_version: str = "1.0"
    """Schema version for future migrations."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    """Timestamp when the run was created."""

    plan: PlanManifest
    """The plan manifest used for this run."""

    run_config: RunConfig
    """The run configuration used for this run."""

    scorecards: list[ScorecardConfig] | None = None
    """Scorecard configurations used for this run, if any."""

    extras: dict[str, Any] = Field(default_factory=dict)
    """Arbitrary extra metadata."""


class ManifestIO:
    """Handle serialization of plan and run manifests."""

    def __init__(self, paths: Paths | None = None) -> None:
        self._paths = paths or Paths.default()

    def plan_path(self, run_id: str) -> Path:
        """Return the filesystem path for the plan manifest of ``run_id``."""

        return self._paths.plan_manifest_path(run_id)

    def plan_revision_path(self, run_id: str, revision: int) -> Path:
        """Return the path for a revision-specific copy of the plan manifest."""

        filename = f"plan_manifest.rev{revision}.json"
        return self._paths.run_dir(run_id) / filename

    def run_path(self, run_id: str) -> Path:
        """Return the filesystem path for the run manifest of ``run_id``."""

        return self._paths.run_manifest_path(run_id)

    def write_plan(
        self,
        run_id: str,
        manifest: PlanManifest,
        *,
        revision: int | None = None,
    ) -> Path:
        """Write a plan manifest to disk and return the canonical path.

        When ``revision`` is provided, an additional revision-specific copy is
        persisted alongside the canonical manifest to preserve history.
        """

        path = self.plan_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = manifest.model_dump_json(indent=4)
        path.write_text(payload, encoding="utf-8")
        if revision is not None:
            rev_path = self.plan_revision_path(run_id, revision)
            rev_path.write_text(payload, encoding="utf-8")
        return path

    def load_plan(self, run_id: str) -> PlanManifest:
        """Load a plan manifest from disk."""

        path = self.plan_path(run_id)
        data = self._read_json(path)
        return PlanManifest.model_validate(data)

    def write_run(self, run_id: str, manifest: RunManifest) -> Path:
        """Write a run manifest to disk and return the path."""

        path = self.run_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(manifest.model_dump_json(indent=4), encoding="utf-8")
        return path

    def load_run(self, run_id: str) -> RunManifest:
        """Load a run manifest from disk."""

        path = self.run_path(run_id)
        data = self._read_json(path)
        return RunManifest.model_validate(data)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cast(dict[str, Any], data)


__all__ = ["ManifestIO", "RunManifest"]
