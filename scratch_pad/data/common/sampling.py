"""Sampling helpers for building balanced subsets."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class StratumStats:
    key: Tuple[str, ...]
    population: int
    target: int


def stratified_sample(
    records: Sequence[T],
    stratify_fn: Callable[[T], Tuple[str, ...]],
    max_samples: int,
    *,
    rng: random.Random | None = None,
    min_per_stratum: int = 1,
) -> List[T]:
    """Return up to ``max_samples`` records using proportional stratified sampling."""
    if rng is None:
        rng = random.Random(0)
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")

    strata_to_records: Dict[Tuple[str, ...], List[T]] = defaultdict(list)
    for record in records:
        key = stratify_fn(record)
        strata_to_records[key].append(record)

    if not strata_to_records:
        raise ValueError("No records to sample from")

    total_population = sum(len(bucket) for bucket in strata_to_records.values())
    take = min(max_samples, total_population)

    stats: Dict[Tuple[str, ...], StratumStats] = {}
    remaining = take
    effective_min = min_per_stratum if take >= len(strata_to_records) else 0

    for key, bucket in strata_to_records.items():
        proportion = len(bucket) / total_population
        allocation = max(effective_min, math.floor(proportion * take))
        allocation = min(allocation, len(bucket))
        stats[key] = StratumStats(key=key, population=len(bucket), target=allocation)
        remaining -= allocation

    if remaining < 0:
        # Reduce allocations when min constraints overshoot the budget.
        deficit = -remaining
        strata = sorted(stats.values(), key=lambda s: s.target, reverse=True)
        for entry in strata:
            if deficit <= 0:
                break
            reducible = min(entry.target, deficit)
            entry.target -= reducible
            deficit -= reducible
        remaining = 0

    if remaining > 0:
        strata = list(stats.values())
        weights = [s.population - s.target for s in strata]
        for _ in range(remaining):
            eligible = [s for s, weight in zip(strata, weights) if weight > 0]
            if not eligible:
                break
            chosen = rng.choice(eligible)
            chosen.target += 1
            idx = strata.index(chosen)
            weights[idx] -= 1

    sampled: List[T] = []
    for key, bucket in strata_to_records.items():
        k = stats[key].target
        if k <= 0:
            continue
        rng.shuffle(bucket)
        sampled.extend(bucket[:k])

    rng.shuffle(sampled)
    return sampled
