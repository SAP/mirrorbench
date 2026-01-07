"""Bootstrap utilities for metric aggregation."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, pstdev


@dataclass(slots=True)
class BootstrapConfig:
    """Configuration for bootstrap resampling."""

    iterations: int = 1000
    confidence: float = 0.95
    random_seed: int | None = None


@dataclass(slots=True)
class BootstrapResult:
    """Summary statistics produced by bootstrap resampling."""

    mean: float
    standard_deviation: float
    ci_lower: float
    ci_upper: float


def bootstrap_mean(values: Sequence[float], config: BootstrapConfig) -> BootstrapResult:
    """Estimate mean statistics via bootstrap resampling."""

    count = len(values)
    if count == 0 or config.iterations <= 0:
        return BootstrapResult(mean=0.0, standard_deviation=0.0, ci_lower=0.0, ci_upper=0.0)

    rng = random.Random(config.random_seed)
    samples: list[float] = []
    population = list(values)

    for _ in range(config.iterations):
        resample = [population[rng.randrange(0, count)] for _ in range(count)]
        samples.append(mean(resample))

    samples.sort()
    mean_estimate = mean(samples)
    std_estimate = pstdev(samples) if len(samples) > 1 else 0.0

    alpha = 1.0 - config.confidence
    lower_idx = max(0, int((alpha / 2.0) * config.iterations))
    upper_idx = min(config.iterations - 1, int((1.0 - alpha / 2.0) * config.iterations) - 1)
    ci_lower = samples[lower_idx]
    ci_upper = samples[upper_idx]

    return BootstrapResult(
        mean=mean_estimate,
        standard_deviation=std_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )
