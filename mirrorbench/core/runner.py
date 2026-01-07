"""High-level runner facade coordinating planners, controllers, and backends."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import structlog

import mirrorbench.core.executor.async_backend  # - ensure builtin backends register
import mirrorbench.core.executor.sync_backend  # noqa: F401 - ensure builtin backends register
from mirrorbench.core.config import JobConfig, RunConfig, ScorecardConfig
from mirrorbench.core.constants import REGISTRY_GROUP_USER_PROXIES, STATUS_COMPLETED
from mirrorbench.core.executor import RunController
from mirrorbench.core.executor.backend_registry import resolve_backend
from mirrorbench.core.manifest import ManifestIO, RunManifest
from mirrorbench.core.models.errors import PlannerError, RunnerError
from mirrorbench.core.models.plan import EvalUnit, PlanManifest
from mirrorbench.core.models.run import RunSummary
from mirrorbench.core.plan import Planner
from mirrorbench.core.registry import registry
from mirrorbench.core.run_db import RunDatabase
from mirrorbench.core.run_db.sqlite import SQLiteRunDatabase
from mirrorbench.io.paths import Paths

_LOG = structlog.get_logger(__name__)


class Backend(Protocol):
    def run(
        self,
        units: Iterable[EvalUnit],
        unit_executor: Callable[[EvalUnit, RunController], None] | None = None,
        *,
        cancel_callback: Callable[[], bool] | None = None,
        progress_callback: Callable[[EvalUnit], None] | None = None,
    ) -> RunSummary: ...


@dataclass(slots=True)
class _DiffResult:
    units_to_run: list[EvalUnit]
    completed_units: set[str]
    missing_units: set[str]


class Runner:
    """Facade that executes plan manifests using the configured backend."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        run_id: str,
        plan_manifest: PlanManifest,
        run_config: RunConfig,
        controller: RunController,
        backend: Backend,
        paths: Paths,
        units_to_run: Sequence[EvalUnit],
        resume: bool,
        revision: int,
        skipped_completed: set[str],
    ) -> None:
        self._run_id = run_id
        self._plan_manifest = plan_manifest
        self._run_config = run_config
        self._controller = controller
        self._backend = backend
        self._paths = paths
        self._units_to_run = list(units_to_run)
        self._resume = resume
        self._revision = revision
        self._skipped_completed = skipped_completed
        self._ran = False
        self._logger = _LOG.bind(run_id=run_id, revision=revision)
        if skipped_completed:
            self._logger.info(
                "resume_skip_completed_units",
                skipped=len(skipped_completed),
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_job_config(  # noqa: PLR0913
        cls,
        job_cfg: JobConfig,
        *,
        run_id: str | None = None,
        paths: Paths | None = None,
        backend_name: str | None = None,
        resume: bool = False,
        scorecards: Sequence[ScorecardConfig] | None = None,
    ) -> Runner:
        """Create a runner from a validated job configuration."""

        resolved_paths = paths or Paths.default()
        if resume and not run_id:
            msg = "resume requires an explicit run_id"
            raise RunnerError(msg)

        planner = Planner.from_config(job_cfg)
        planner.build()
        if planner.manifest is None:
            raise PlannerError("Planner did not produce a manifest")

        resolved_run_id = run_id or resolved_paths.new_run_id()
        scorecards = scorecards if scorecards is not None else job_cfg.scorecards or []

        return cls.from_plan(
            planner.manifest,
            run_config=job_cfg.run,
            run_id=resolved_run_id,
            paths=resolved_paths,
            backend_name=backend_name,
            resume=resume,
            scorecards=scorecards,
        )

    @classmethod
    def from_plan(  # noqa: PLR0913
        cls,
        plan_manifest: PlanManifest,
        *,
        run_config: RunConfig,
        run_id: str,
        paths: Paths | None = None,
        backend_name: str | None = None,
        resume: bool = False,
        scorecards: Sequence[ScorecardConfig] | None = None,
    ) -> Runner:
        """Create a runner from an explicit plan manifest."""

        if not run_id:
            raise RunnerError("run_id must not be empty")

        resolved_paths = paths or Paths.default()
        manifest_io = ManifestIO(resolved_paths)
        run_dir = resolved_paths.runs_dir() / run_id
        run_exists = run_dir.exists()

        if run_exists and not resume:
            raise RunnerError(
                f"Run '{run_id}' already exists. Pass resume=True to append a revision."
            )
        if resume and not run_exists:
            raise RunnerError(f"Run '{run_id}' does not exist; cannot resume.")

        previous_manifest: RunManifest | None = None
        if run_exists:
            try:
                previous_manifest = manifest_io.load_run(run_id)
            except FileNotFoundError:
                previous_manifest = None

        revision = 0
        if previous_manifest is not None:
            prev_revision = previous_manifest.extras.get("revision", 0)
            try:
                revision = int(prev_revision) + 1
            except (TypeError, ValueError):
                revision = 1
        elif resume:
            # Resume requested but no prior manifest (e.g., interrupted before completion)
            revision = 1

        db: RunDatabase = SQLiteRunDatabase(resolved_paths.run_db_path(run_id), run_id)
        db.initialize()
        existing_statuses = db.get_unit_statuses()

        executed_history: set[str] = set()
        if previous_manifest is not None:
            executed_payload = previous_manifest.extras.get("executed_units")
            if isinstance(executed_payload, list):
                executed_history = {str(item) for item in executed_payload}

        diff = cls._diff_units(plan_manifest, existing_statuses, executed_history)
        if diff.missing_units:
            missing = ", ".join(sorted(diff.missing_units))
            raise RunnerError(f"New manifest is missing previously recorded units: {missing}")

        backend_key = backend_name or run_config.engine
        inherited_scorecards: Sequence[ScorecardConfig] | None = scorecards
        if inherited_scorecards is None and previous_manifest is not None:
            inherited_scorecards = previous_manifest.scorecards

        controller = RunController(
            run_id=run_id,
            plan_manifest=plan_manifest,
            run_config=run_config,
            paths=resolved_paths,
            db=db,
            manifest_io=manifest_io,
            scorecards=list(inherited_scorecards or []),
            revision=revision,
            previous_run_manifest=previous_manifest,
        )
        backend = cls._select_backend(backend_key, controller)

        runner = cls(
            run_id=run_id,
            plan_manifest=plan_manifest,
            run_config=run_config,
            controller=controller,
            backend=backend,
            paths=resolved_paths,
            units_to_run=diff.units_to_run,
            resume=resume,
            revision=revision,
            skipped_completed=diff.completed_units,
        )
        return runner

    def validate_runtime(self) -> None:
        """Ensure runtime dependencies (credentials, SDKs) are available before execution."""

        for proxy_spec in self._plan_manifest.user_proxies:
            adapter_name = proxy_spec.adapter or proxy_spec.name
            try:
                adapter_factory = registry.factory(REGISTRY_GROUP_USER_PROXIES, adapter_name)
            except Exception as exc:  # pragma: no cover - defensive
                raise RunnerError(f"Failed to resolve user proxy adapter '{adapter_name}'") from exc

            adapter = adapter_factory()
            configure_cache = getattr(adapter, "configure_cache", None)
            if callable(configure_cache):
                configure_cache(cache_config=self._run_config.cache, paths=self._paths)

            try:
                adapter.validate_runtime(config=proxy_spec, run_id=self._run_id)
            except Exception as exc:  # - propagate meaningful message
                raise RunnerError(
                    f"User proxy '{proxy_spec.name}' failed credential or dependency validation"
                ) from exc

    @staticmethod
    def _diff_units(
        manifest: PlanManifest,
        existing_statuses: dict[str, str],
        executed_history: set[str] | None = None,
    ) -> _DiffResult:
        units_to_run: list[EvalUnit] = []
        completed_units: set[str] = set()
        missing_units: set[str] = set()

        status_map = dict(existing_statuses)
        history = set(executed_history or set())
        seen_existing = set(status_map.keys()) | history

        for unit in manifest.units:
            unit_id = unit.unit_id()
            status = status_map.get(unit_id)
            if status is None:
                if unit_id in history:
                    completed_units.add(unit_id)
                else:
                    units_to_run.append(unit)
                seen_existing.discard(unit_id)
                continue

            if status != STATUS_COMPLETED:
                units_to_run.append(unit)
            else:
                completed_units.add(unit_id)
            seen_existing.discard(unit_id)

        if seen_existing:
            missing_units = seen_existing

        return _DiffResult(
            units_to_run=units_to_run, completed_units=completed_units, missing_units=missing_units
        )

    @staticmethod
    def _select_backend(name: str, controller: RunController) -> Backend:
        normalized = name.strip().lower()
        try:
            backend_factory = resolve_backend(normalized)
        except KeyError:
            alias_name = normalized
            if alias_name in {"ray", "distributed"}:
                raise RunnerError("Ray execution backend is not implemented yet") from None
            raise RunnerError(f"Unknown execution backend '{name}'") from None
        backend = backend_factory(controller)
        return cast(Backend, backend)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def run_id(self) -> str:
        return self._run_id

    def run(
        self,
        *,
        cancel_callback: Callable[[], bool] | None = None,
        progress_callback: Callable[[EvalUnit], None] | None = None,
    ) -> RunSummary:
        """Execute the manifest and return the final run summary."""

        if self._ran:
            raise RunnerError("Runner.run() may only be called once per instance")
        self._logger.info(
            "runner_execute",
            planned_units=len(self._plan_manifest.units),
            units_to_run=len(self._units_to_run),
            resume=self._resume,
        )
        try:
            summary = self._backend.run(
                self._units_to_run,
                cancel_callback=cancel_callback,
                progress_callback=progress_callback,
            )
        finally:
            self._ran = True
        return summary

    def dry_run(self) -> RunController:
        """Return the controller for inspection without executing any units."""

        self._logger.info("runner_dry_run")
        return self._controller

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def planned_units(self) -> list[EvalUnit]:
        """Expose the manifest units (including completed ones)."""

        return list(self._plan_manifest.units)

    def pending_units(self) -> list[EvalUnit]:
        """Expose the subset of units scheduled for execution in this revision."""

        return list(self._units_to_run)

    def skipped_completed_units(self) -> set[str]:
        """Return unit identifiers skipped because results already exist."""

        return set(self._skipped_completed)

    def path(self) -> Path:
        """Return the filesystem path where run artifacts are stored."""

        return self._paths.run_dir(self._run_id)
