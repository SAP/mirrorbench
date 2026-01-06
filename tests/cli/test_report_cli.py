from __future__ import annotations

import json
from types import SimpleNamespace

from mirrorbench.cli import cmd_report_json
from mirrorbench.io.paths import Paths


def test_report_json_stdout(tmp_path, capsys, monkeypatch) -> None:
    paths = Paths(tmp_path / "mirrorbench")
    run_id = "run-001"
    summary = {
        "run": {"run_id": run_id, "extra": {"status": "completed"}},
        "aggregates": [],
        "units": [],
        "telemetry_stats": {},
        "notes": [],
    }
    paths.save_run_summary(run_id, summary)
    monkeypatch.setattr("mirrorbench.cli.Paths.default", lambda: paths)

    args = SimpleNamespace(run_id=run_id, output=None)
    cmd_report_json(args)
    output = json.loads(capsys.readouterr().out)
    assert output["run"]["run_id"] == run_id


def test_report_json_writes_file(tmp_path, monkeypatch) -> None:
    paths = Paths(tmp_path / "mirrorbench")
    run_id = "run-002"
    summary = {
        "run": {"run_id": run_id, "extra": {"status": "completed"}},
        "aggregates": [],
        "units": [],
        "telemetry_stats": {},
        "notes": [],
    }
    paths.save_run_summary(run_id, summary)
    monkeypatch.setattr("mirrorbench.cli.Paths.default", lambda: paths)

    output_path = tmp_path / "report.json"
    args = SimpleNamespace(run_id=run_id, output=str(output_path))
    cmd_report_json(args)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["run"]["run_id"] == run_id
