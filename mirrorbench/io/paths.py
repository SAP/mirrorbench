import json
import time
import uuid
from pathlib import Path
from typing import Any, cast

from platformdirs import PlatformDirs

from mirrorbench.core.constants import (
    APP_AUTHOR,
    APP_NAME,
    CACHE_DB_FILENAME,
    CACHE_DIRNAME,
    DATASETS_DIRNAME,
    LAST_RUN_TEXT_FILENAME,
    PLAN_MANIFEST_FILENAME,
    RUN_DB_FILENAME,
    RUN_MANIFEST_FILENAME,
    RUNS_DIRNAME,
    SUMMARY_FILENAME,
)


class Paths:
    """Filesystem helper for storing MirrorBench artifacts under user directories."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.runs_dir().mkdir(parents=True, exist_ok=True)
        self.cache_dir()

    @classmethod
    def default(cls) -> "Paths":
        d = PlatformDirs(appname=APP_NAME, appauthor=APP_AUTHOR)
        base = Path(d.user_data_dir)
        base.mkdir(parents=True, exist_ok=True)
        return cls(base)

    def runs_dir(self) -> Path:
        """Return the root directory where run artifacts are stored."""

        return self.base / RUNS_DIRNAME

    def datasets_dir(self) -> Path:
        """Return the root directory where dataset artifacts are cached."""

        path = self.base / DATASETS_DIRNAME
        path.mkdir(parents=True, exist_ok=True)
        return path

    def dataset_root(self, dataset_name: str) -> Path:
        """Return the directory reserved for a specific dataset name."""

        path = self.datasets_dir() / dataset_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def dataset_cache_dir(self, dataset_name: str, cache_key: str) -> Path:
        """Return the directory for a dataset cache key, creating it if needed."""

        path = self.dataset_root(dataset_name) / cache_key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_dir(self, run_id: str) -> Path:
        """Return the directory for a specific run, creating it if needed."""

        path = self.runs_dir() / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_manifest_path(self, run_id: str) -> Path:
        """Path to the persisted run manifest JSON file."""

        return self.run_dir(run_id) / RUN_MANIFEST_FILENAME

    def plan_manifest_path(self, run_id: str) -> Path:
        """Path to the persisted plan manifest JSON file."""

        return self.run_dir(run_id) / PLAN_MANIFEST_FILENAME

    def summary_path(self, run_id: str) -> Path:
        """Path to the persisted run summary JSON file."""

        return self.run_dir(run_id) / SUMMARY_FILENAME

    def run_db_path(self, run_id: str) -> Path:
        """Path to the SQLite database file for a run."""

        return self.run_dir(run_id) / RUN_DB_FILENAME

    def cache_dir(self) -> Path:
        """Return the directory storing cache artefacts."""

        path = self.base / CACHE_DIRNAME
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cache_db_path(self) -> Path:
        """Return the path to the shared cache database."""

        return self.cache_dir() / CACHE_DB_FILENAME

    def new_run_id(self) -> str:
        rid = time.strftime("%Y%m%d-%H%M%S") + "-" + str(uuid.uuid4())[:8]
        (self.base / LAST_RUN_TEXT_FILENAME).write_text(rid, encoding="utf-8")
        return rid

    def load_last_run_id(self) -> str | None:
        f = self.base / LAST_RUN_TEXT_FILENAME
        return f.read_text(encoding="utf-8").strip() if f.exists() else None

    def save_run_summary(self, run_id: str, summary: dict[str, Any]) -> None:
        summary_path = self.summary_path(run_id)
        summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    def load_run_summary(self, run_id: str) -> dict[str, Any]:
        summary_path = self.summary_path(run_id)
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        return cast(dict[str, Any], data)
