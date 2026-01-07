"""Project-wide constants used across MirrorBench."""

from typing import Final, Literal, TypeAlias

APP_NAME: Final = "mirrorbench"
APP_AUTHOR: Final = "mirrorbench"

LAST_RUN_TEXT_FILENAME: Final = "last_run.txt"
RUNS_DIRNAME: Final = "runs"
DATASETS_DIRNAME: Final = "datasets"
PLAN_MANIFEST_FILENAME: Final = "plan.json"
RUN_MANIFEST_FILENAME: Final = "manifest.json"
SUMMARY_FILENAME: Final = "summary.json"
RUN_DB_FILENAME: Final = "run.db"
DEFAULT_RETRY_BACKOFF_SECONDS: Final = 2.0
CACHE_DIRNAME: Final = "cache"
CACHE_DB_FILENAME: Final = "cache.db"
DEFAULT_CACHE_TTL_SECONDS = 86_400

DEFAULT_SCORECARD_NAME: Final = "mirror_scorecard"
DEFAULT_TASK_DRIVER_NAME: Final = "task:default/single_turn"

OPENAI_API_KEY_ENV: Final = "OPENAI_API_KEY"
OPENAI_ORG_ID_ENV: Final = "OPENAI_ORG_ID"
HF_TOKEN_ENV: Final = "HF_TOKEN"

STATUS_CREATED: Final = "created"
STATUS_PENDING: Final = "pending"
STATUS_RUNNING: Final = "running"
STATUS_COMPLETED: Final = "completed"
STATUS_FAILED: Final = "failed"
STATUS_CANCELLED: Final = "cancelled"
STATUSES_ALL: Final[tuple[str, ...]] = (
    STATUS_CREATED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_CANCELLED,
)

RegistryGroupName: TypeAlias = Literal[  # noqa: UP040
    "user_proxies",
    "datasets",
    "metrics",
    "model_clients",
    "tasks",
    "judges",
]

REGISTRY_GROUP_USER_PROXIES: Final[RegistryGroupName] = "user_proxies"
REGISTRY_GROUP_DATASETS: Final[RegistryGroupName] = "datasets"
REGISTRY_GROUP_METRICS: Final[RegistryGroupName] = "metrics"
REGISTRY_GROUP_MODEL_CLIENTS: Final[RegistryGroupName] = "model_clients"
REGISTRY_GROUP_TASKS: Final[RegistryGroupName] = "tasks"
REGISTRY_GROUP_JUDGES: Final[RegistryGroupName] = "judges"
REGISTRY_GROUPS: Final[tuple[RegistryGroupName, ...]] = (
    REGISTRY_GROUP_USER_PROXIES,
    REGISTRY_GROUP_DATASETS,
    REGISTRY_GROUP_METRICS,
    REGISTRY_GROUP_MODEL_CLIENTS,
    REGISTRY_GROUP_TASKS,
    REGISTRY_GROUP_JUDGES,
)


# ruff: noqa: RUF022
__all__ = [
    "APP_AUTHOR",
    "APP_NAME",
    "DEFAULT_SCORECARD_NAME",
    "OPENAI_API_KEY_ENV",
    "OPENAI_ORG_ID_ENV",
    "HF_TOKEN_ENV",
    "STATUS_CREATED",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_CANCELLED",
    "STATUSES_ALL",
    "LAST_RUN_TEXT_FILENAME",
    "DATASETS_DIRNAME",
    "PLAN_MANIFEST_FILENAME",
    "RUN_DB_FILENAME",
    "RUN_MANIFEST_FILENAME",
    "RUNS_DIRNAME",
    "SUMMARY_FILENAME",
    "CACHE_DIRNAME",
    "CACHE_DB_FILENAME",
    "DEFAULT_TASK_DRIVER_NAME",
    "RegistryGroupName",
    "REGISTRY_GROUP_USER_PROXIES",
    "REGISTRY_GROUP_DATASETS",
    "REGISTRY_GROUP_METRICS",
    "REGISTRY_GROUP_MODEL_CLIENTS",
    "REGISTRY_GROUP_TASKS",
    "REGISTRY_GROUP_JUDGES",
    "REGISTRY_GROUPS",
]
