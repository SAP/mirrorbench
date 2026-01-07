"""Pydantic models and helpers for job configuration validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from mirrorbench.core.constants import (
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_RETRY_BACKOFF_SECONDS,
    DEFAULT_SCORECARD_NAME,
)
from mirrorbench.core.models.plan import DatasetSpec, JudgeSpec, MetricSpec, UserProxySpec


class ObservabilityConfig(BaseModel):
    """Logging and telemetry configuration."""

    log_json: bool = Field(True, description="Emit logs as JSON when True, otherwise text")
    log_level: str = Field("INFO", description="Logging level (DEBUG, INFO, etc.)")
    log_destination: str | None = Field(
        default=None,
        description="Optional file path for log output (stdout when omitted)",
    )
    tracing_enabled: bool = Field(False, description="Enable OpenTelemetry tracing")
    metrics_enabled: bool = Field(False, description="Enable OpenTelemetry metrics")
    otel_exporter: str = Field("noop", description="Telemetry exporter (noop|stdout|otlp)")
    otel_endpoint: str | None = Field(
        default=None,
        description="Endpoint for OTLP exporter",
    )

    @model_validator(mode="after")
    def normalise_exporter(self) -> ObservabilityConfig:
        exporter = self.otel_exporter.lower()
        if exporter not in {"noop", "stdout", "otlp"}:
            raise ValueError("otel_exporter must be one of: noop, stdout, otlp")
        self.otel_exporter = exporter
        return self


def _default_observability_config() -> ObservabilityConfig:
    return ObservabilityConfig.model_validate({})


class CacheConfig(BaseModel):
    """Configuration for cache behaviour."""

    enabled: bool = Field(True, description="Whether caching is enabled")
    ttl_seconds: int | None = Field(
        default=DEFAULT_CACHE_TTL_SECONDS,
        ge=0,
        description="Default TTL for cache entries in seconds (0/None = no expiry)",
    )
    backend: str = Field("sqlite", description="Cache backend identifier")

    @model_validator(mode="after")
    def normalise(self) -> CacheConfig:
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            self.ttl_seconds = None
        return self


class BootstrapOptions(BaseModel):
    """Bootstrap configuration applied to metric aggregates."""

    iterations: int = Field(1000, ge=0, description="Number of bootstrap resamples")
    confidence: float = Field(
        0.95,
        gt=0.0,
        lt=1.0,
        description="Confidence level used for intervals (0 < value < 1)",
    )


class MetricsRuntimeConfig(BaseModel):
    """Runtime options applied to metrics during aggregation."""

    bootstrap: BootstrapOptions | None = Field(
        default=None,
        description="Optional bootstrap settings applied to metrics lacking explicit params.",
    )


class RunConfig(BaseModel):
    """Execution options parsed from the job configuration."""

    name: str = Field("mirrorbench_run", description="Logical run name")
    """Name of the run."""

    seeds: list[int] = Field(default_factory=lambda: [0], description="Seeds used for permutation")
    """List of integer seeds for evaluation unit permutations."""

    engine: str = Field("sync", description="Execution backend identifier")
    """Identifier for the execution backend to use."""

    max_concurrency: int = Field(8, ge=1, description="Maximum evaluation units in flight")
    """Maximum number of evaluation units to execute concurrently."""

    timeout_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Per-unit timeout in seconds (None disables timeouts)",
    )
    """Timeout applied to each evaluation unit."""

    max_retries: int = Field(0, ge=0, description="Number of retries for failed units")
    """Number of retries when a unit fails due to an exception or timeout."""

    retry_backoff_seconds: float = Field(
        DEFAULT_RETRY_BACKOFF_SECONDS,
        ge=0,
        description="Base backoff (seconds) between retries; multiplied by attempt index",
    )
    """Base backoff interval between retry attempts."""

    cache: CacheConfig = Field(
        default_factory=lambda: CacheConfig(
            enabled=True,
            ttl_seconds=DEFAULT_CACHE_TTL_SECONDS,
            backend="sqlite",
        ),
        description="Cache configuration for model and judge calls.",
    )
    """Cache configuration."""

    judge: JudgeSpec | None = Field(None, description="Legacy default judge (per-metric preferred)")
    """Optional default judge specification (overridden by per-metric judges)."""

    observability: ObservabilityConfig = Field(
        default_factory=_default_observability_config,
        description="Logging and telemetry configuration.",
    )
    """Observability settings for the run."""

    metrics: MetricsRuntimeConfig = Field(
        default_factory=MetricsRuntimeConfig,
        description="Metric runtime configuration (bootstrap, etc.).",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_cache(cls, values: dict[str, Any]) -> dict[str, Any]:
        cache_value = values.get("cache")
        if isinstance(cache_value, bool):
            values["cache"] = {"enabled": cache_value}
        return values

    @model_validator(mode="after")
    def ensure_seeds(self) -> RunConfig:
        """Ensure at least one seed is provided."""

        if not self.seeds:
            raise ValueError("run.seeds must contain at least one seed")
        if self.timeout_seconds is not None and self.timeout_seconds == 0:
            self.timeout_seconds = None
        return self


class ScorecardConfig(BaseModel):
    """Configuration for aggregating metrics into a scorecard."""

    name: str = Field(DEFAULT_SCORECARD_NAME, description="Name of the scorecard")
    """Name of the scorecard."""

    weights: dict[str, float] = Field(default_factory=dict, description="Metric weights")
    """Mapping of metric names to their weights in the scorecard."""

    @model_validator(mode="after")
    def ensure_weights(self) -> ScorecardConfig:
        if not self.weights:
            raise ValueError("scorecard weights must not be empty")
        total = float(sum(self.weights.values()))
        if total <= 0:
            raise ValueError("scorecard weights must sum to a positive value")
        self.weights = {k: float(v) / total for k, v in self.weights.items()}
        return self


class TaskDriverOverride(BaseModel):
    """Job-level override for dataset task driver selection and parameters."""

    driver: str | None = Field(
        default=None,
        description="Optional explicit task driver name for the dataset",
    )
    """Identifier for the task driver to use (overrides dataset default)."""

    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters forwarded to the resolved task driver",
    )
    """Parameters to pass to the task driver."""

    @model_validator(mode="after")
    def validate_params(self) -> TaskDriverOverride:
        if not isinstance(self.params, dict):
            raise ValueError("task driver params must be a mapping")
        return self


class JobConfig(BaseModel):
    """Validated representation of a job configuration file."""

    run: RunConfig
    """Execution configuration for the job."""

    user_proxies: list[UserProxySpec]
    """List of user proxies to evaluate."""

    datasets: list[DatasetSpec]
    """List of datasets against which to evaluate user proxies."""

    metrics: list[MetricSpec]
    """List of metrics to compute for each evaluation unit."""

    scorecards: list[ScorecardConfig] | None = Field(
        default=None,
        description="Optional scorecard configurations providing metric weights.",
    )

    task_drivers: dict[str, TaskDriverOverride] = Field(
        default_factory=dict,
        description="Optional per-dataset task driver overrides.",
    )

    @model_validator(mode="after")
    def ensure_lists(self) -> JobConfig:
        """Ensure each top-level collection is non-empty."""

        if not self.user_proxies:
            raise ValueError("At least one user proxy must be configured")
        if not self.datasets:
            raise ValueError("At least one dataset must be configured")
        if not self.metrics:
            raise ValueError("At least one metric must be configured")
        if self.scorecards is None:
            weights = {metric.name: 1.0 / len(self.metrics) for metric in self.metrics}
            self.scorecards = [ScorecardConfig(name=DEFAULT_SCORECARD_NAME, weights=weights)]
        dataset_names = {dataset.name for dataset in self.datasets}
        unknown_overrides = set(self.task_drivers) - dataset_names
        if unknown_overrides:
            unknown = ", ".join(sorted(unknown_overrides))
            raise ValueError(f"Task driver overrides reference unknown datasets: {unknown}")
        return self


def load_job_config(path: str | Path) -> JobConfig:
    """Read a YAML/JSON config file and validate it into a :class:`JobConfig`."""

    raw = Path(path).read_text(encoding="utf-8")
    data = cast(dict[str, Any], yaml.safe_load(raw))
    try:
        return JobConfig.model_validate(data)
    except ValidationError as exc:  # pragma: no cover - reporting path
        # re-raise with readable message for CLI consumers
        raise SystemExit(f"Invalid job configuration: {exc}") from exc
