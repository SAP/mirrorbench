"""Metadata models describing registered adapters, datasets, metrics, and judges."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class _MetadataModel(BaseModel):
    """Base helper enabling plugin-extensible metadata."""

    model_config = ConfigDict(extra="allow", frozen=True)


class UserProxyAdapterInfo(_MetadataModel):
    """Metadata describing a user proxy adapter implementation."""

    name: str
    """Unique name of the adapter."""

    capabilities: set[str] = Field(default_factory=set)
    """Set of capability tags supported by this adapter."""

    supported_tasks: set[str] = Field(default_factory=set)
    """Set of task tags supported by this adapter."""

    provider: str | None = None
    """Optional provider name (e.g., 'openai', 'azure', 'anthropic')."""

    concurrency_limit: int | None = None
    """Optional max concurrency limit for this adapter."""

    description: str | None = None
    """Optional human-readable description of the adapter."""

    pricing: dict[str, Any] | None = None
    """Optional pricing metadata (e.g., cost per token)."""

    default_rate_limits: dict[str, Any] | None = None
    """Optional default rate limit metadata (requests per minute, tokens)."""


class DatasetInfo(_MetadataModel):
    """Metadata describing a dataset loader."""

    name: str
    """Unique name of the dataset."""

    supported_tasks: set[str] = Field(default_factory=set)
    """Set of task tags supported by this dataset."""

    splits: set[str] = Field(default_factory=set)
    """Set of available data splits (e.g., 'train', 'validation', 'test')."""

    languages: set[str] = Field(default_factory=set)
    """Set of languages present in the dataset (e.g., 'en', 'fr')."""

    reference_stats_mode: Literal["bundled", "generate", "external", "optional"] = "optional"
    """How reference statistics are obtained (bundled, generated, external, optional)."""

    citation: str | None = None
    """Optional citation or URL for the dataset."""


class MetricInfo(_MetadataModel):
    """Metadata describing a metric implementation."""

    name: str
    """Unique name of the metric."""

    needs_references: bool = False
    """Whether the metric requires reference outputs."""

    needs_judge: bool = False
    """Whether the metric requires an LLM-as-a-judge."""

    supported_tasks: set[str] = Field(default_factory=set)
    """Set of task tags supported by this metric."""

    category: str | None = None
    """Optional category of the metric (e.g., 'relevance', 'toxicity')."""


class JudgeInfo(_MetadataModel):
    """Metadata describing an LLM-as-a-judge implementation."""

    name: str
    """Unique name of the judge."""

    model: str | None = None
    """Optional model name or identifier used by the judge."""

    prompt_version: str | None = None
    """Optional version of the prompt used by the judge."""

    rubric: str | None = None
    """Optional human-readable description of the judge's rubric."""


class ModelClientInfo(_MetadataModel):
    """Metadata describing a model client implementation."""

    name: str
    """Unique registry name for the client (e.g., ``client:openai/chat``)."""

    provider: str
    """Provider identifier (e.g., ``openai``, ``langchain``)."""

    capabilities: set[str] = Field(default_factory=set)
    """Set of capability tags supported by this client (e.g., ``chat``, ``streaming``)."""

    models: set[str] | None = None
    """Optional set of provider model identifiers supported by this client."""

    default_rate_limits: dict[str, float] | None = None
    """Optional rate-limit hints (requests/minute, tokens/minute)."""

    pricing: dict[str, Any] | None = None
    """Optional pricing metadata."""

    telemetry_keys: set[str] | None = None
    """Telemetry fields emitted by the client (e.g., ``tokens_input``, ``time_to_first_token``)."""


class TaskDriverInfo(_MetadataModel):
    """Metadata describing a task driver implementation."""

    name: str
    """Unique name of the task driver (e.g., ``task:mirror/conversation``)."""

    supported_tasks: set[str] = Field(default_factory=set)
    """Set of task tags handled by the driver (values from ``EpisodeSpec.task_tag``)."""

    description: str | None = None
    """Optional human-readable description of the driver."""

    default_dataset_names: set[str] = Field(default_factory=set)
    """Datasets typically using this driver (informational only)."""


__all__ = [
    "DatasetInfo",
    "JudgeInfo",
    "MetricInfo",
    "ModelClientInfo",
    "TaskDriverInfo",
    "UserProxyAdapterInfo",
]
