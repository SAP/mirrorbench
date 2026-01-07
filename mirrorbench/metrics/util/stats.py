"""Statistical helpers for metric aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from statistics import NormalDist, mean, stdev

_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

_DEFAULT_T_CONFIDENCE = 0.95


def mean_stdev_ci(
    values: Sequence[float],
    *,
    confidence: float = _DEFAULT_T_CONFIDENCE,
) -> tuple[float, float, float]:
    """Return mean, standard deviation, and half-width CI for numeric values."""

    count = len(values)
    if count == 0:
        return 0.0, 0.0, 0.0

    mean_value = mean(values)
    stdev_value = stdev(values) if count > 1 else 0.0

    if count <= 1 or stdev_value == 0.0:
        return mean_value, stdev_value, 0.0

    alpha = 1.0 - confidence
    if confidence == _DEFAULT_T_CONFIDENCE and count - 1 in _T_CRITICAL_95:
        critical = _T_CRITICAL_95[count - 1]
    else:
        dist = NormalDist()
        critical = dist.inv_cdf(1.0 - alpha / 2.0)

    half_width = critical * stdev_value / sqrt(count)
    return mean_value, stdev_value, half_width
