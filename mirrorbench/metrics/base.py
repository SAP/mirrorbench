"""Shared utilities and mixins for metric implementations."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean
from typing import Any

from mirrorbench.core.models.run import MetricAggregate, MetricValue
from mirrorbench.metrics.util.stats import mean_stdev_ci


def default_metric_aggregate(
    values: Sequence[MetricValue],
    *,
    metric_name: str,
) -> MetricAggregate:
    """Compute default aggregation (mean ± stdev) for per-episode metric values.

    This is the default aggregation strategy used by BaseMetric.aggregate().
    Metrics can override aggregate() to provide custom aggregation logic
    (e.g., bootstrap CIs, harmonic mean, etc.).

    Args:
        values: Sequence of per-episode metric values.
        metric_name: Name of the metric being aggregated.

    Returns:
        MetricAggregate containing mean, standard deviation, and sample size.
    """
    all_values: list[float] = []
    for value in values:
        all_values.extend(value.values)

    if not all_values:
        return MetricAggregate(
            metric_name=metric_name,
            mean=0.0,
            standard_deviation=0.0,
            confidence_interval=0.0,
            sample_size=0,
        )

    mean_value, stdev_value, ci_half_width = mean_stdev_ci(all_values)

    return MetricAggregate(
        metric_name=metric_name,
        mean=mean_value,
        standard_deviation=stdev_value,
        confidence_interval=ci_half_width,
        sample_size=len(all_values),
    )


def combine_metric_values(
    values: Sequence[MetricValue],
    *,
    operation: str = "mean",
) -> float:
    """Combine multiple metric values using the specified operation.

    Helper for metrics that need to reduce per-episode values before
    returning a final aggregate.

    Args:
        values: Sequence of metric values to combine.
        operation: Aggregation operation ("mean", "sum", "max", "min").

    Returns:
        Combined scalar value.

    Raises:
        ValueError: If operation is not recognized or values are empty.
    """
    if not values:
        msg = "Cannot combine empty sequence of metric values"
        raise ValueError(msg)

    all_floats: list[float] = []
    for value in values:
        all_floats.extend(value.values)

    if not all_floats:
        msg = "No numeric values found in metric values"
        raise ValueError(msg)

    if operation == "mean":
        return float(mean(all_floats))
    if operation == "sum":
        return float(sum(all_floats))
    if operation == "max":
        return float(max(all_floats))
    if operation == "min":
        return float(min(all_floats))

    msg = f"Unknown operation: {operation}"
    raise ValueError(msg)


def extract_metadata_field(
    values: Sequence[MetricValue],
    field: str,
    *,
    default: Any = None,
) -> list[Any]:
    """Extract a metadata field from all metric values.

    Useful when metrics need to aggregate metadata across episodes
    (e.g., collecting all token counts for corpus-level statistics).

    Args:
        values: Sequence of metric values.
        field: Metadata field name to extract.
        default: Default value if field is missing.

    Returns:
        List of extracted field values (one per MetricValue).
    """
    result: list[Any] = []
    for value in values:
        result.append(value.metadata.get(field, default))
    return result


__all__ = [
    "combine_metric_values",
    "default_metric_aggregate",
    "extract_metadata_field",
]
