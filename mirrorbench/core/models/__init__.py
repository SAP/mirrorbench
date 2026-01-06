"""Core data models underpinning planner, runner, and reporting layers."""

from mirrorbench.core.models.cache import CacheBackend, CacheEntry, CacheError, CacheKey, CacheStats
from mirrorbench.core.models.episodes import (
    EpisodeArtifact,
    EpisodeLog,
    EpisodeSpec,
    ReferenceStats,
)
from mirrorbench.core.models.errors import (
    ConfigError,
    DatasetError,
    JudgeError,
    MetricError,
    MirrorBenchError,
    PlannerError,
    RegistryError,
    RunnerError,
)
from mirrorbench.core.models.messages import JudgeVerdict, Message, Role, TurnTelemetry
from mirrorbench.core.models.plan import (
    DatasetSpec,
    EvalUnit,
    JudgeSpec,
    MetricSpec,
    Plan,
    PlanManifest,
    SkipRecord,
    UserProxySpec,
)
from mirrorbench.core.models.registry import (
    DatasetInfo,
    JudgeInfo,
    MetricInfo,
    ModelClientInfo,
    UserProxyAdapterInfo,
)
from mirrorbench.core.models.run import (
    EpisodeResult,
    MetricAggregate,
    MetricValue,
    RunMetadata,
    RunSummary,
)
from mirrorbench.core.registry.entries import RegistryEntry

# ruff: noqa: RUF022
__all__ = [
    "ConfigError",
    "CacheBackend",
    "CacheEntry",
    "CacheError",
    "CacheKey",
    "CacheStats",
    "DatasetError",
    "DatasetInfo",
    "DatasetSpec",
    "EpisodeArtifact",
    "EpisodeLog",
    "EpisodeResult",
    "EpisodeSpec",
    "EvalUnit",
    "JudgeError",
    "JudgeInfo",
    "JudgeSpec",
    "JudgeVerdict",
    "Message",
    "ModelClientInfo",
    "MetricAggregate",
    "MetricError",
    "MetricInfo",
    "MetricSpec",
    "MetricValue",
    "MirrorBenchError",
    "Plan",
    "PlanManifest",
    "PlannerError",
    "ReferenceStats",
    "RegistryEntry",
    "RegistryError",
    "Role",
    "RunMetadata",
    "RunSummary",
    "RunnerError",
    "SkipRecord",
    "TurnTelemetry",
    "UserProxyAdapterInfo",
    "UserProxySpec",
]
