"""Metric implementations and shared utilities.

Importing this module triggers registration of all built-in metrics with
the global registry, making them discoverable via the planner and CLI.
"""

from __future__ import annotations

# Import registry to trigger auto-registration of built-in metrics
import mirrorbench.metrics.registry  # noqa: F401

# Re-export key types and utilities for convenience
from mirrorbench.metrics.base import (
    combine_metric_values,
    default_metric_aggregate,
    extract_metadata_field,
)
from mirrorbench.metrics.judge.critique_then_revise import CritiqueThenReviseMetric
from mirrorbench.metrics.judge.gteval import GTEvalMetric
from mirrorbench.metrics.judge.pairwise_indistinguishability import (
    PairwiseIndistinguishabilityMetric,
)
from mirrorbench.metrics.judge.rubric_and_reason import RubricAndReasonMetric
from mirrorbench.metrics.lexical.type_token_ratio import TypeTokenRatioMetric
from mirrorbench.metrics.util import (
    DEFAULT_REFERENCE_KEYS,
    compute_token_ngrams,
    distinct_n,
    resolve_reference_conversation,
    tokenize,
)

# ruff: noqa: RUF022
__all__ = [
    # Metrics
    "TypeTokenRatioMetric",
    "CritiqueThenReviseMetric",
    "GTEvalMetric",
    "PairwiseIndistinguishabilityMetric",
    "RubricAndReasonMetric",
    # Base utilities
    "default_metric_aggregate",
    "combine_metric_values",
    "extract_metadata_field",
    # Conversation/text utilities
    "DEFAULT_REFERENCE_KEYS",
    "resolve_reference_conversation",
    "tokenize",
    "compute_token_ngrams",
    "distinct_n",
]
