"""Moving-Average Type-Token Ratio (MATTR) metric."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from mirrorbench.core.models.registry import MetricInfo
from mirrorbench.core.registry.decorators import register_metric
from mirrorbench.metrics.lexical._base import BaseLexicalDiversityMetric

METRIC_NAME = "metric:lexical/mattr"

MATTR_METRIC_INFO = MetricInfo(
    name=METRIC_NAME,
    needs_references=True,
    needs_judge=False,
    supported_tasks=set(),
    category="lexical_diversity",
)


def _mattr(tokens: Sequence[int], window: int) -> float:
    """Compute moving-average TTR over a fixed window."""

    length = len(tokens)
    if length == 0:
        return 0.0
    if window <= 0:
        raise ValueError("window must be positive")
    if length <= window:
        return len(set(tokens)) / length

    type_counts: dict[int, int] = {}
    unique = 0
    for token in tokens[:window]:
        type_counts[token] = type_counts.get(token, 0) + 1
        if type_counts[token] == 1:
            unique += 1
    ttr_sum = unique / window

    for idx in range(window, length):
        outgoing = tokens[idx - window]
        type_counts[outgoing] -= 1
        if type_counts[outgoing] == 0:
            unique -= 1
            del type_counts[outgoing]
        incoming = tokens[idx]
        type_counts[incoming] = type_counts.get(incoming, 0) + 1
        if type_counts[incoming] == 1:
            unique += 1
        ttr_sum += unique / window

    return ttr_sum / (length - window + 1)


@register_metric(name=METRIC_NAME, metadata=MATTR_METRIC_INFO)
class MATTRMetric(BaseLexicalDiversityMetric):
    """MirrorBench MATTR implementation with human-anchored normalization."""

    info: ClassVar[MetricInfo] = MATTR_METRIC_INFO

    def __init__(
        self,
        *,
        window: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = window

    def _score(self, tokens: Sequence[int]) -> float:
        return _mattr(tokens, self.window)

    def _params_metadata(self) -> dict[str, int]:
        return {"window": self.window}


__all__ = ["MATTRMetric", "MATTR_METRIC_INFO", "METRIC_NAME"]
