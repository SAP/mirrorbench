"""Planning-time configuration models and execution units."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class _BaseSpec(BaseModel):
    """Shared behaviour for normalized configuration specs."""

    model_config = ConfigDict(extra="allow", frozen=True)


class UserProxySpec(_BaseSpec):
    """Normalized definition of a user proxy adapter."""

    name: str
    """Name of the proxy, e.g. `gpt-4`, `claude-3`, `my-custom-agent`, etc."""

    adapter: str | None = None
    """Optional adapter name."""

    params: dict[str, Any] = Field(default_factory=dict)
    """Arbitrary parameters passed to the proxy adapter."""

    label: str | None = None
    """Optional human-readable label for reports and visualizations."""


class DatasetSpec(_BaseSpec):
    """Normalized definition of a dataset entry in a job configuration."""

    name: str
    """Name of the dataset, e.g. `sft-belle`, `gpt4all-instruct`, etc."""

    split: str = "default"
    """Dataset split to use (mirror datasets ship a `default` split by convention)."""

    params: dict[str, Any] = Field(default_factory=dict)
    """Arbitrary parameters passed to the dataset loader."""

    task: str | None = None
    """Optional task tag to filter dataset episodes."""

    label: str | None = None
    """Optional human-friendly dataset label for reporting."""


class JudgeSpec(_BaseSpec):
    """Judge configuration used by a metric."""

    name: str
    """Name of the judge."""

    params: dict[str, Any] = Field(default_factory=dict)
    """Arbitrary parameters passed to the judge implementation."""

    label: str | None = None
    """Optional display name for judge outputs."""


class MetricSpec(_BaseSpec):
    """Normalized definition of a metric configuration."""

    name: str
    """Name of the metric, e.g. `bleu`, `rouge`, `bert-score`, `gpt-4-eval`, etc."""

    params: dict[str, Any] = Field(default_factory=dict)
    """Arbitrary parameters passed to the metric implementation."""

    label: str | None = None
    """Optional alias used in scorecards and charts."""

    judge: JudgeSpec | None = None
    """Judge configuration required by this metric, if any."""


class TaskDriverSpec(_BaseSpec):
    """Resolved task driver configuration for a dataset."""

    name: str
    """Registry name of the task driver."""

    params: dict[str, Any] = Field(default_factory=dict)
    """Driver-specific parameters (dataset defaults merged with job overrides)."""


class SkipRecord(BaseModel):
    """Describe why a particular proxy/dataset/metric combination was skipped."""

    proxy: str
    """Name of the proxy that was skipped."""

    dataset: str
    """Name of the dataset that was skipped."""

    metric: str
    """Name of the metric that was skipped."""

    reason: str
    """Human-readable reason for skipping the combination."""


class EvalUnit(BaseModel):
    """One evaluation unit executed by the runner."""

    proxy_name: str
    """Name of the proxy to evaluate."""

    dataset_name: str
    """Name of the dataset to draw episodes from."""

    dataset_split: str = "default"
    """Dataset split to use (e.g., 'train', 'test', 'default')."""

    metric_name: str
    """Name of the metric to compute."""

    seed: int = 0
    """Random seed used for sampling episodes, if applicable."""

    judge_name: str | None = None
    """Optional judge name, if the metric requires a judge."""

    def unit_id(self) -> str:
        """Deterministic identifier used for persistence and telemetry."""

        return f"{self.proxy_name}|{self.dataset_name}|{self.dataset_split}|{self.metric_name}|{self.seed}"


@dataclass(slots=True)
class Plan:
    """Collection of :class:`EvalUnit` objects scheduled for execution."""

    units: list[EvalUnit] = field(default_factory=list)
    """List of evaluation units to run."""


class PlanManifest(BaseModel):
    """Full manifest emitted by the planner for reproducibility."""

    user_proxies: list[UserProxySpec] = Field(default_factory=list)
    """List of user proxies to evaluate."""

    datasets: list[DatasetSpec] = Field(default_factory=list)
    """List of datasets to draw episodes from."""

    metrics: list[MetricSpec] = Field(default_factory=list)
    """List of metrics to compute."""

    metric_judges: dict[str, JudgeSpec] = Field(default_factory=dict)
    """Resolved judge configuration per metric."""

    task_drivers: dict[str, TaskDriverSpec] = Field(default_factory=dict)
    """Resolved task driver per dataset (keyed by dataset name)."""

    config_hash: str = ""
    """Deterministic hash of the validated job configuration."""

    units: list[EvalUnit] = Field(default_factory=list)
    """List of evaluation units to run."""

    skipped: list[SkipRecord] = Field(default_factory=list)
    """List of skipped proxy/dataset/metric combinations."""

    planner_version: str = "0.0.1"
    """Version of the planner that generated this manifest."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    """Timestamp when the manifest was created."""

    def to_plan(self) -> Plan:
        """Convert the manifest into a lightweight :class:`Plan`."""

        return Plan(units=list(self.units))
