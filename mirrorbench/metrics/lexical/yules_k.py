"""Yule's K lexical diversity metric."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, ClassVar

from mirrorbench.core.models.registry import MetricInfo
from mirrorbench.core.registry.decorators import register_metric
from mirrorbench.metrics.lexical._base import BaseLexicalDiversityMetric

METRIC_NAME = "metric:lexical/yules_k"

YULES_K_METRIC_INFO = MetricInfo(
    name=METRIC_NAME,
    needs_references=True,
    needs_judge=False,
    supported_tasks=set(),
    category="lexical_diversity",
)


def _yules_k(tokens: Sequence[int]) -> float:
    """Compute Yule's characteristic constant from token frequencies."""

    counts = Counter(tokens)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    sum_sq = sum(freq * freq for freq in counts.values())
    return 10_000.0 * (sum_sq - total) / (total * total)


@register_metric(name=METRIC_NAME, metadata=YULES_K_METRIC_INFO)
class YulesKMetric(BaseLexicalDiversityMetric):
    """MirrorBench implementation of Yule's K lexical diversity metric."""

    info: ClassVar[MetricInfo] = YULES_K_METRIC_INFO

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

    def _score(self, tokens: Sequence[int]) -> float:
        return _yules_k(tokens)

    def _params_metadata(self) -> dict[str, Any]:
        return {}


__all__ = ["YULES_K_METRIC_INFO", "YulesKMetric", "METRIC_NAME"]
