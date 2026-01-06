"""SQLite schema definitions for run database."""

SCHEMA_VERSION = 2

CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT DEFAULT 'created',
    engine TEXT,
    planner_version TEXT,
    summary_json TEXT,
    notes TEXT
);
"""

CREATE_UNITS = """
CREATE TABLE IF NOT EXISTS units (
    run_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    user_proxy TEXT NOT NULL,
    dataset TEXT NOT NULL,
    dataset_split TEXT NOT NULL DEFAULT 'default',
    metric TEXT NOT NULL,
    seed INTEGER NOT NULL,
    judge TEXT,
    status TEXT DEFAULT 'pending',
    PRIMARY KEY (run_id, unit_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
"""

CREATE_EPISODES = """
CREATE TABLE IF NOT EXISTS episodes (
    run_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_s REAL,
    artifact_path TEXT,
    summary TEXT,
    metric_values TEXT,
    telemetry_json TEXT,
    PRIMARY KEY (run_id, unit_id, episode_id),
    FOREIGN KEY (run_id, unit_id) REFERENCES units(run_id, unit_id) ON DELETE CASCADE
);
"""

CREATE_METRICS = """
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    mean REAL NOT NULL,
    standard_deviation REAL,
    confidence_interval REAL,
    p_value REAL,
    sample_size INTEGER NOT NULL,
    extras TEXT,
    PRIMARY KEY (run_id, unit_id, metric),
    FOREIGN KEY (run_id, unit_id) REFERENCES units(run_id, unit_id) ON DELETE CASCADE
);
"""

CREATE_TELEMETRY = """
CREATE TABLE IF NOT EXISTS telemetry (
    run_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (run_id, unit_id, key),
    FOREIGN KEY (run_id, unit_id) REFERENCES units(run_id, unit_id) ON DELETE CASCADE
);
"""

CREATE_SCORECARDS = """
CREATE TABLE IF NOT EXISTS scorecards (
    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    score REAL,
    weights TEXT NOT NULL,
    missing_metrics TEXT,
    extras TEXT,
    PRIMARY KEY (run_id, name),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_units_run ON units(run_id);",
    "CREATE INDEX IF NOT EXISTS idx_episodes_run_unit ON episodes(run_id, unit_id);",
    "CREATE INDEX IF NOT EXISTS idx_metrics_run_unit ON metrics(run_id, unit_id);",
]

CREATE_TABLE_STATEMENTS = [
    CREATE_RUNS,
    CREATE_UNITS,
    CREATE_EPISODES,
    CREATE_METRICS,
    CREATE_TELEMETRY,
    CREATE_SCORECARDS,
]
