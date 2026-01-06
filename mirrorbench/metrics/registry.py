"""Register built-in metrics with the global registry."""

from __future__ import annotations

from mirrorbench.metrics.judge.gteval import GTEvalMetric
from mirrorbench.metrics.lexical.hdd import HDDMetric
from mirrorbench.metrics.lexical.mattr import MATTRMetric
from mirrorbench.metrics.lexical.type_token_ratio import TypeTokenRatioMetric
from mirrorbench.metrics.lexical.yules_k import YulesKMetric

# ruff: noqa: RUF022
__all__ = [
    "TypeTokenRatioMetric",
    "GTEvalMetric",
    "MATTRMetric",
    "HDDMetric",
    "YulesKMetric",
]
