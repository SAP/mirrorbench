"""Hypergeometric distribution diversity (HD-D) metric."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import comb
from typing import Any, ClassVar

from mirrorbench.core.models.registry import MetricInfo
from mirrorbench.core.registry.decorators import register_metric
from mirrorbench.metrics.lexical._base import BaseLexicalDiversityMetric

METRIC_NAME = "metric:lexical/hdd"

HDD_METRIC_INFO = MetricInfo(
    name=METRIC_NAME,
    needs_references=True,
    needs_judge=False,
    supported_tasks=set(),
    category="lexical_diversity",
)


def _hdd(tokens: list[int], sample_size: int) -> float:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    population = len(tokens)
    if population == 0:
        return 0.0

    sample = min(sample_size, population)
    denominator = comb(population, sample)
    frequencies = Counter(tokens)
    diversity = 0.0
    for count in frequencies.values():
        diversity += 1.0 - (comb(population - count, sample) / denominator)
    return diversity / sample


@register_metric(name=METRIC_NAME, metadata=HDD_METRIC_INFO)
class HDDMetric(BaseLexicalDiversityMetric):
    """Human-anchored HD-D metric."""

    info: ClassVar[MetricInfo] = HDD_METRIC_INFO

    def __init__(
        self,
        *,
        sample_size: int = 42,
        **kwargs: Any,
    ) -> None:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        super().__init__(**kwargs)
        self.sample_size = sample_size

    def _score(self, tokens: Sequence[int]) -> float:
        return _hdd(list(tokens), self.sample_size)

    def _params_metadata(self) -> dict[str, int]:
        return {"sample_size": self.sample_size}


__all__ = ["HDDMetric", "HDD_METRIC_INFO", "METRIC_NAME"]
