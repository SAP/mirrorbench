from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from mirrorbench.cli import cmd_runs_delete, cmd_runs_inspect
from mirrorbench.core.constants import LAST_RUN_TEXT_FILENAME
from mirrorbench.core.run_db import schema
from mirrorbench.io.paths import Paths


def test_runs_delete_removes_directory(tmp_path, monkeypatch, capsys) -> None:
    paths = Paths(tmp_path / "mirrorbench")
    run_id = "run-delete"
    run_dir = paths.run_dir(run_id)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    (paths.base / LAST_RUN_TEXT_FILENAME).write_text(run_id, encoding="utf-8")

    monkeypatch.setattr("mirrorbench.cli.Paths.default", lambda: paths)

    args = SimpleNamespace(run_id=run_id, force=True)
    cmd_runs_delete(args)

    assert not run_dir.exists()
    assert not (paths.base / LAST_RUN_TEXT_FILENAME).exists()
    out = capsys.readouterr().out
    assert f"deleted run '{run_id}'" in out


def test_runs_delete_requires_force(tmp_path, monkeypatch) -> None:
    paths = Paths(tmp_path / "mirrorbench")
    run_id = "run-no-force"
    _ = paths.run_dir(run_id)

    monkeypatch.setattr("mirrorbench.cli.Paths.default", lambda: paths)

    args = SimpleNamespace(run_id=run_id, force=False)
    with pytest.raises(SystemExit):
        cmd_runs_delete(args)

    assert (paths.runs_dir() / run_id).exists()


def test_runs_delete_missing_run(tmp_path, monkeypatch) -> None:
    paths = Paths(tmp_path / "mirrorbench")
    monkeypatch.setattr("mirrorbench.cli.Paths.default", lambda: paths)

    args = SimpleNamespace(run_id="does-not-exist", force=True)
    with pytest.raises(SystemExit):
        cmd_runs_delete(args)


def test_runs_inspect_outputs_episode(tmp_path, monkeypatch, capsys) -> None:
    paths = Paths(tmp_path / "mirrorbench")
    run_id = "run-inspect"
    run_dir = paths.run_dir(run_id)
    db_path = paths.run_db_path(run_id)

    with sqlite3.connect(db_path) as conn:
        for statement in schema.CREATE_TABLE_STATEMENTS:
            conn.executescript(statement)
        conn.execute(
            "INSERT INTO runs (run_id, created_at, status, engine, planner_version)"
            " VALUES (?, datetime('now'), 'completed', 'sync', '0.1')",
            (run_id,),
        )
        conn.execute(
            "INSERT INTO units (run_id, unit_id, user_proxy, dataset, metric, seed, judge, status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, "proxy|data|metric|0", "proxy", "data", "metric", 0, None, "completed"),
        )
        conn.execute(
            "INSERT INTO episodes (run_id, unit_id, episode_id, status, artifact_path, metric_values, telemetry_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                "proxy|data|metric|0",
                "ep-001",
                "completed",
                "artifacts/proxy|data|metric|0/ep-001.json",
                json.dumps({"metric": {"values": [], "metadata": {}}}),
                json.dumps({"tokens_input": 5}),
            ),
        )

    artifact_file = run_dir / "artifacts" / "proxy|data|metric|0" / "ep-001.json"
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_file.write_text(json.dumps({"spec": {"episode_id": "ep-001"}}), encoding="utf-8")

    monkeypatch.setattr("mirrorbench.cli.Paths.default", lambda: paths)

    args = SimpleNamespace(
        run_id=run_id,
        unit_id=None,
        episode_id=None,
        index=0,
        output=None,
    )
    cmd_runs_inspect(args)
    output = json.loads(capsys.readouterr().out)
    assert output["unit_id"] == "proxy|data|metric|0"
    assert output["episode_id"] == "ep-001"
    assert output["artifact"]["spec"]["episode_id"] == "ep-001"


def test_runs_inspect_no_match(tmp_path, monkeypatch) -> None:
    paths = Paths(tmp_path / "mirrorbench")
    run_id = "run-empty"
    paths.run_dir(run_id)  # ensure directory exists
    db_path = paths.run_db_path(run_id)
    with sqlite3.connect(db_path) as conn:
        for statement in schema.CREATE_TABLE_STATEMENTS:
            conn.executescript(statement)

    monkeypatch.setattr("mirrorbench.cli.Paths.default", lambda: paths)

    args = SimpleNamespace(run_id=run_id, unit_id=None, episode_id=None, index=0, output=None)
    with pytest.raises(SystemExit):
        cmd_runs_inspect(args)
