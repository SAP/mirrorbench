"""MirrorBench CLI providing access to planning and execution utilities."""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import signal
import sqlite3
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

import mirrorbench  # noqa: F401 - trigger registry imports
from mirrorbench.cache import get_cache_manager
from mirrorbench.core.config import (
    CacheConfig,
    JobConfig,
    load_job_config,
)
from mirrorbench.core.constants import (
    DEFAULT_CACHE_TTL_SECONDS,
    LAST_RUN_TEXT_FILENAME,
    RUN_MANIFEST_FILENAME,
)
from mirrorbench.core.logging import configure_logging_from_config
from mirrorbench.core.models.errors import RunnerError
from mirrorbench.core.models.plan import EvalUnit
from mirrorbench.core.plan import Planner
from mirrorbench.core.runner import Runner
from mirrorbench.core.telemetry import configure_telemetry_from_config
from mirrorbench.io.paths import Paths
from mirrorbench.report import generate_json_report

PREVIEW_LIMIT = 20


def _load_config(path: str) -> JobConfig:
    """Load and validate a job configuration file."""

    return load_job_config(path)


def cmd_plan(args: argparse.Namespace) -> None:
    job_cfg = _load_config(args.config)
    configure_logging_from_config(job_cfg.run.observability)
    configure_telemetry_from_config(job_cfg.run.observability)
    plan = Planner.from_config(job_cfg).build()
    print(f"[mirrorbench] Plan contains {len(plan.units)} units:")
    for unit in plan.units[:PREVIEW_LIMIT]:
        print(
            f"  - (proxy={unit.proxy_name}, dataset={unit.dataset_name}, "
            f"metric={unit.metric_name}, seed={unit.seed})"
        )
    remaining = len(plan.units) - PREVIEW_LIMIT
    if remaining > 0:
        print(f"  ... (+{remaining} more)")


def _not_implemented(command: str) -> None:
    msg = (
        f"[mirrorbench] '{command}' is not implemented yet. "
        "Refer to ADR-0001 for details on the upcoming execution engine."
    )
    print(msg, file=sys.stderr)
    sys.exit(1)


def _resolve_run_id(paths: Paths, run_id: str | None, resume: bool) -> str:
    if resume:
        if run_id:
            return run_id
        previous = paths.load_last_run_id()
        if previous:
            return previous
        raise RunnerError("No previous runs available to resume.")
    return run_id or paths.new_run_id()


def _resolve_report_run_id(paths: Paths, run_id: str) -> str:
    if run_id == "last":
        previous = paths.load_last_run_id()
        if not previous:
            print("No previous runs found.", file=sys.stderr)
            raise SystemExit(1)
        return previous
    return run_id


def cmd_dryrun(args: argparse.Namespace) -> None:
    job_cfg = _load_config(args.config)
    paths = Paths.default()
    run_id = _resolve_run_id(paths, args.run_id, args.resume)
    runner = Runner.from_job_config(
        job_cfg,
        run_id=run_id,
        paths=paths,
        backend_name=args.backend,
        resume=args.resume,
    )

    runner.validate_runtime()

    controller = runner.dry_run()
    planned = len(runner.planned_units())
    pending = len(runner.pending_units())
    manifest_path = paths.plan_manifest_path(run_id)
    print(f"[mirrorbench] Dry run prepared for {run_id}.")
    print(f"  Planned units: {planned}")
    print(f"  Pending units for execution: {pending}")
    print(f"  Plan manifest written to: {manifest_path}")
    controller.close()


def cmd_run(args: argparse.Namespace) -> None:  # noqa: PLR0915
    job_cfg = _load_config(args.config)
    paths = Paths.default()
    run_id = _resolve_run_id(paths, args.run_id or job_cfg.run.name, args.resume)

    obs = job_cfg.run.observability.model_copy()
    if getattr(args, "log_level", None):
        obs.log_level = args.log_level
    if getattr(args, "log_json", False):
        obs.log_json = True
    if getattr(args, "log_text", False):
        obs.log_json = False
    if getattr(args, "log_destination", None):
        obs.log_destination = args.log_destination
    if getattr(args, "otel_exporter", None):
        obs.otel_exporter = args.otel_exporter
    if getattr(args, "otel_endpoint", None):
        obs.otel_endpoint = args.otel_endpoint
    if getattr(args, "enable_tracing", False):
        obs.tracing_enabled = True
    if getattr(args, "enable_metrics", False):
        obs.metrics_enabled = True

    job_cfg.run.observability = obs
    configure_logging_from_config(obs)
    configure_telemetry_from_config(obs)

    runner = Runner.from_job_config(
        job_cfg,
        run_id=run_id,
        paths=paths,
        backend_name=args.backend,
        resume=args.resume,
    )

    runner.validate_runtime()

    total_units = len(runner.planned_units())
    completed = 0
    cancel_requested = False

    def progress_callback(unit: EvalUnit) -> None:
        nonlocal completed
        completed += 1
        if not args.no_progress:
            print(
                f"[mirrorbench] completed {completed}/{total_units}: {unit.unit_id()}",
                flush=True,
            )

    def cancel_callback() -> bool:
        return cancel_requested

    def handle_sigint(signum: int, frame: object) -> None:  # pragma: no cover - signal handling
        nonlocal cancel_requested
        cancel_requested = True
        print(
            "\n[mirrorbench] Cancellation requested. Waiting for current unit to finish...",
            file=sys.stderr,
        )

    old_handler = signal.signal(signal.SIGINT, handle_sigint)
    try:
        summary = runner.run(
            cancel_callback=cancel_callback,
            progress_callback=progress_callback,
        )
    finally:
        signal.signal(signal.SIGINT, old_handler)

    summary_data = paths.load_run_summary(run_id)
    status = summary.run.extra.get("status")
    episodes_successful = summary.run.extra.get("episodes_successful", 0)
    episodes_failed = summary.run.extra.get("episodes_failed", 0)
    episodes_total = summary.run.extra.get("episodes_total", 0)
    units_completed = summary.run.extra.get("units_completed", 0)
    units_total = summary.run.extra.get("units_total", 0)

    print(f"[mirrorbench] Run {run_id} finished with status: {status}")
    print(f"[mirrorbench] Units: {units_completed}/{units_total} completed")
    print(
        f"[mirrorbench] Episodes: {episodes_successful}/{episodes_total} successful, {episodes_failed} failed"
    )
    print(json.dumps(summary_data, indent=2))


def cmd_report(args: argparse.Namespace) -> None:
    raise SystemExit("Specify a subcommand, e.g. 'mirrorbench report json <run_id>'")


def cmd_report_json(args: argparse.Namespace) -> None:
    paths = Paths.default()
    run_id = _resolve_report_run_id(paths, args.run_id)
    summary = paths.load_run_summary(run_id)
    report = generate_json_report(summary)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[mirrorbench] JSON report written to {output_path}")
    else:
        print(json.dumps(report, indent=2))


def cmd_cache_stats(args: argparse.Namespace) -> None:
    paths = Paths.default()
    manager = get_cache_manager(
        paths,
        CacheConfig(
            enabled=True,
            ttl_seconds=DEFAULT_CACHE_TTL_SECONDS,
            backend="sqlite",
        ),
    )
    stats = manager.stats(namespace=args.namespace)
    if not stats:
        print("[mirrorbench] cache is empty.")
        return
    print("namespace\tentries\ttotal_bytes\toldest\tnewest")
    for item in stats:
        oldest = item.oldest_entry.isoformat() if item.oldest_entry else "-"
        newest = item.newest_entry.isoformat() if item.newest_entry else "-"
        print(f"{item.namespace}\t{item.entries}\t{item.total_bytes}\t{oldest}\t{newest}")


def cmd_cache_purge(args: argparse.Namespace) -> None:
    paths = Paths.default()
    manager = get_cache_manager(
        paths,
        CacheConfig(
            enabled=True,
            ttl_seconds=DEFAULT_CACHE_TTL_SECONDS,
            backend="sqlite",
        ),
    )
    removed = manager.purge(namespace=args.namespace)
    ns = args.namespace or "all namespaces"
    print(f"[mirrorbench] purged {removed} entries from {ns}.")


def cmd_runs_delete(args: argparse.Namespace) -> None:
    paths = Paths.default()
    run_dir = paths.runs_dir() / args.run_id
    if not run_dir.exists():
        print(f"[mirrorbench] run '{args.run_id}' does not exist.", file=sys.stderr)
        raise SystemExit(1)

    if not args.force:
        print(
            "[mirrorbench] refusing to delete run without --force."
            " Re-run with --force to confirm removal.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    shutil.rmtree(run_dir)

    last_run_file = paths.base / LAST_RUN_TEXT_FILENAME
    if last_run_file.exists() and last_run_file.read_text(encoding="utf-8").strip() == args.run_id:
        last_run_file.unlink()

    print(f"[mirrorbench] deleted run '{args.run_id}'.")


def cmd_runs_inspect(args: argparse.Namespace) -> None:
    paths = Paths.default()
    db_path = paths.run_db_path(args.run_id)
    if not db_path.exists():
        print(f"[mirrorbench] run '{args.run_id}' does not exist.", file=sys.stderr)
        raise SystemExit(1)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        query = (
            "SELECT unit_id, episode_id, status, duration_s, artifact_path, "
            "metric_values, telemetry_json FROM episodes WHERE run_id = ?"
        )
        params: list[Any] = [args.run_id]
        if args.unit_id:
            query += " AND unit_id = ?"
            params.append(args.unit_id)
        if args.episode_id:
            query += " AND episode_id = ?"
            params.append(args.episode_id)
        query += " ORDER BY unit_id, episode_id LIMIT 1 OFFSET ?"
        params.append(max(args.index, 0))

        row = conn.execute(query, params).fetchone()

    if row is None:
        print("[mirrorbench] no matching episodes found.", file=sys.stderr)
        raise SystemExit(1)

    result = {
        "run_id": args.run_id,
        "unit_id": row["unit_id"],
        "episode_id": row["episode_id"],
        "status": row["status"],
        "duration_s": row["duration_s"],
        "metrics": json.loads(row["metric_values"]) if row["metric_values"] else {},
        "telemetry": json.loads(row["telemetry_json"]) if row["telemetry_json"] else {},
        "artifact_path": row["artifact_path"],
        "artifact": None,
    }

    if row["artifact_path"]:
        artifact_file = paths.run_dir(args.run_id) / row["artifact_path"]
        if artifact_file.exists():
            result["artifact"] = json.loads(artifact_file.read_text(encoding="utf-8"))

    payload = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        print(f"[mirrorbench] episode artifact written to {output_path}")
    else:
        print(payload)


def cmd_runs_delete_from_configs(args: argparse.Namespace) -> None:  # noqa: PLR0912, PLR0915
    """Delete all runs corresponding to config YAML files in the configs directory."""

    paths = Paths.default()
    configs_dir = Path(args.configs_dir)

    if not configs_dir.exists():
        print(f"[mirrorbench] configs directory '{configs_dir}' does not exist.", file=sys.stderr)
        raise SystemExit(1)

    # Find all YAML config files
    config_files = list(configs_dir.glob("**/*.yaml")) + list(configs_dir.glob("**/*.yml"))

    if not config_files:
        print(f"[mirrorbench] no config files found in '{configs_dir}'.", file=sys.stderr)
        raise SystemExit(0)

    # Extract run names from config files
    run_names_to_delete: set[str] = set()
    for config_file in config_files:
        try:
            job_cfg = _load_config(str(config_file))
            run_name = job_cfg.run.name
            if run_name:
                run_names_to_delete.add(run_name)
        except Exception as exc:
            print(
                f"[mirrorbench] warning: failed to load config '{config_file}': {exc}",
                file=sys.stderr,
            )
            continue

    if not run_names_to_delete:
        print("[mirrorbench] no valid run names found in config files.", file=sys.stderr)
        raise SystemExit(0)

    print(f"[mirrorbench] found {len(run_names_to_delete)} unique run names from config files.")

    # Find all existing runs
    runs_dir = paths.runs_dir()
    if not runs_dir.exists():
        print("[mirrorbench] no runs directory found.", file=sys.stderr)
        raise SystemExit(0)

    # Match run directories with config run names
    runs_to_delete: list[str] = []

    for run_name in run_names_to_delete:
        # Find run directories that match the run name pattern
        matching_runs = list(runs_dir.glob(f"*{run_name}*"))

        for run_dir in matching_runs:
            run_id = run_dir.name

            # Verify this is actually a run directory by checking for run manifest
            run_manifest_path = run_dir / RUN_MANIFEST_FILENAME
            if not run_manifest_path.exists():
                continue

            # Load run manifest to verify the name matches
            try:
                with open(run_manifest_path, encoding="utf-8") as f:
                    manifest_data = json.load(f)
                    manifest_run_name = manifest_data.get("run_config", {}).get("name")

                    if manifest_run_name == run_name:
                        runs_to_delete.append(run_id)
            except Exception as exc:
                print(
                    f"[mirrorbench] warning: failed to verify run '{run_id}': {exc}",
                    file=sys.stderr,
                )
                continue

    if not runs_to_delete:
        print("[mirrorbench] no matching runs found to delete.", file=sys.stderr)
        raise SystemExit(0)

    print(f"[mirrorbench] found {len(runs_to_delete)} run(s) to delete:")
    for run_id in runs_to_delete:
        print(f"  - {run_id}")

    if not args.force:
        print("[mirrorbench] use --force to confirm deletion.", file=sys.stderr)
        raise SystemExit(0)

    # Delete each run using the existing delete function
    deleted_count = 0
    for run_id in runs_to_delete:
        # Create a mock args object for cmd_runs_delete
        delete_args = argparse.Namespace(run_id=run_id, force=True)
        try:
            cmd_runs_delete(delete_args)
            deleted_count += 1
        except SystemExit:
            # cmd_runs_delete raises SystemExit(1) if run doesn't exist, we can ignore
            pass
        except Exception as exc:
            print(f"[mirrorbench] error deleting run '{run_id}': {exc}", file=sys.stderr)

    print(f"[mirrorbench] successfully deleted {deleted_count} run(s).")


def cmd_runs_list(args: argparse.Namespace) -> None:
    paths = Paths.default()
    runs_dir = paths.runs_dir()

    if not runs_dir.exists():
        print("[mirrorbench] no runs found.")
        return

    runs: list[dict[str, Any]] = []
    for item in runs_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            ts = item.stat().st_ctime
            created = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            runs.append(
                {"name": item.name, "path": str(item.resolve()), "created": created, "ts": ts}
            )

    if not runs:
        print("[mirrorbench] no runs found.")
        return

    runs.sort(key=lambda x: x["ts"], reverse=True)

    try:
        console = Console()
        table = Table(header_style="bold magenta")
        table.add_column("Name", style="cyan")
        table.add_column("Created At", style="green")
        table.add_column("Path", style="dim")

        for run in runs:
            table.add_row(run["name"], run["created"], run["path"])

        console.print(table)

    except ImportError:
        # Fallback if rich is not installed
        print(f"{'Name':<40} {'Created At':<25} {'Path'}")
        print("-" * 100)
        for run in runs:
            print(f"{run['name']:<40} {run['created']:<25} {run['path']}")


def _add_plan_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("-c", "--config", required=True)
    plan_parser.set_defaults(func=cmd_plan)


def _add_dryrun_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    dryrun_parser = sub.add_parser("dryrun")
    dryrun_parser.add_argument("-c", "--config", required=True)
    dryrun_parser.add_argument("--run-id")
    dryrun_parser.add_argument("--backend")
    dryrun_parser.add_argument("--resume", action="store_true")
    dryrun_parser.set_defaults(func=cmd_dryrun)


def _add_run_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run_parser = sub.add_parser("run")
    run_parser.add_argument("-c", "--config", required=True)
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument("--backend")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--no-progress", action="store_true")
    run_parser.add_argument("--log-level")
    run_parser.add_argument("--log-json", action="store_true")
    run_parser.add_argument("--log-text", action="store_true")
    run_parser.add_argument("--log-destination")
    run_parser.add_argument("--enable-tracing", action="store_true")
    run_parser.add_argument("--enable-metrics", action="store_true")
    run_parser.add_argument("--otel-exporter")
    run_parser.add_argument("--otel-endpoint")
    run_parser.set_defaults(func=cmd_run)


def _add_report_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    report_parser = sub.add_parser("report")
    report_sub = report_parser.add_subparsers(dest="report_cmd", required=True)
    report_json = report_sub.add_parser("json")
    report_json.add_argument("run_id")
    report_json.add_argument("--output")
    report_json.set_defaults(func=cmd_report_json)


def _add_cache_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    cache_parser = sub.add_parser("cache")
    cache_sub = cache_parser.add_subparsers(dest="cache_cmd", required=True)
    stats_parser = cache_sub.add_parser("stats")
    stats_parser.add_argument("--namespace")
    stats_parser.set_defaults(func=cmd_cache_stats)
    purge_parser = cache_sub.add_parser("purge")
    purge_parser.add_argument("--namespace")
    purge_parser.set_defaults(func=cmd_cache_purge)


def _add_runs_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    runs_parser = sub.add_parser("runs")
    runs_sub = runs_parser.add_subparsers(dest="runs_cmd", required=True)
    runs_delete = runs_sub.add_parser("delete")
    runs_delete.add_argument("run_id")
    runs_delete.add_argument("-f", "--force", action="store_true")
    runs_delete.set_defaults(func=cmd_runs_delete)
    runs_delete_configs = runs_sub.add_parser("delete-from-configs")
    runs_delete_configs.add_argument(
        "--configs_dir",
        default="configs",
        help="Directory containing config YAML files",
        required=False,
    )
    runs_delete_configs.add_argument("-f", "--force", action="store_true", help="Confirm deletion")
    runs_delete_configs.set_defaults(func=cmd_runs_delete_from_configs)
    runs_inspect = runs_sub.add_parser("inspect")
    runs_inspect.add_argument("run_id")
    runs_inspect.add_argument("--unit-id")
    runs_inspect.add_argument("--episode-id")
    runs_inspect.add_argument("--index", type=int, default=0)
    runs_inspect.add_argument("--output")
    runs_inspect.set_defaults(func=cmd_runs_inspect)
    runs_list = runs_sub.add_parser("list")
    runs_list.set_defaults(func=cmd_runs_list)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mirrorbench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_plan_subparser(sub)
    _add_dryrun_subparser(sub)
    _add_run_subparser(sub)
    _add_report_subparser(sub)
    _add_cache_subparser(sub)
    _add_runs_subparser(sub)

    return parser


def main() -> None:
    parser = _build_parser()

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
