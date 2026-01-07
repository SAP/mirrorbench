"""Structured exception hierarchy used across MirrorBench."""

from __future__ import annotations


class MirrorBenchError(Exception):
    """Base class for all domain-specific errors raised by MirrorBench."""


class ConfigError(MirrorBenchError):
    """Raised when a user-supplied configuration is invalid."""


class PlannerError(MirrorBenchError):
    """Raised when the planner fails to construct a valid plan."""


class RegistryError(MirrorBenchError):
    """Raised when registry lookups or registrations fail."""


class RunnerError(MirrorBenchError):
    """Raised when the execution engine encounters a non-recoverable error."""


class MetricError(MirrorBenchError):
    """Raised when a metric computation fails."""


class DatasetError(MirrorBenchError):
    """Raised when dataset loading or preprocessing fails."""


class JudgeError(MirrorBenchError):
    """Raised when an LLM judge call or normalization fails."""


class TaskDriverError(MirrorBenchError):
    """Raised when task driver resolution or execution fails."""
