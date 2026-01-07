"""Z-score helpers shared by lexical metrics."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean, stdev


def safe_mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean or 0.0 when empty."""

    return mean(values) if values else 0.0


def safe_stdev(values: Sequence[float]) -> float:
    """Return sample standard deviation guarding small samples."""

    if not values or len(values) == 1:
        return 0.0
    return stdev(values)


def safe_z_score(value: float, mean_value: float, stdev_value: float) -> float:
    """Compute z-score while handling zero variance cases."""

    if stdev_value == 0:
        return 0.0
    return (value - mean_value) / stdev_value


__all__ = ["safe_mean", "safe_stdev", "safe_z_score"]
