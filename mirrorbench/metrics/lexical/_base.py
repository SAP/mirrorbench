"""Abstract helpers for lexical diversity metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any, ClassVar, cast

from mirrorbench.core.models.episodes import EpisodeArtifact
from mirrorbench.core.models.messages import Role
from mirrorbench.core.models.registry import MetricInfo
from mirrorbench.core.models.run import MetricAggregate, MetricValue
from mirrorbench.core.registry import BaseMetric
from mirrorbench.metrics.lexical._shared import human_tokens, mean_and_stdev, proxy_tokens
from mirrorbench.metrics.util.stats import mean_stdev_ci
from mirrorbench.metrics.util.zscore import safe_z_score


@dataclass(slots=True)
class LexicalMetricParams:
    """Common configuration passed through metadata/extras."""

    tokenizer_model: str
    target_role: str
    min_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_model": self.tokenizer_model,
            "target_role": self.target_role,
            "min_tokens": self.min_tokens,
        }


class BaseLexicalDiversityMetric(BaseMetric):
    """Base logic shared by MATTR/HD-D/Yule's K metrics."""

    info: ClassVar[MetricInfo]

    def __init__(
        self,
        *,
        tokenizer_model: str = "gpt-4o",
        target_role: Role | str = Role.USER,
        min_tokens: int = 5,
    ) -> None:
        if isinstance(target_role, str):
            target_role = Role(target_role.lower())
        self.tokenizer_model = tokenizer_model
        self.target_role = target_role
        self.min_tokens = min_tokens

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def _score(self, tokens: Sequence[int]) -> float:
        raise NotImplementedError

    def _params_metadata(self) -> dict[str, Any]:
        return {}

    # ------------------------------------------------------------------
    # BaseMetric overrides
    # ------------------------------------------------------------------
    def evaluate(self, episode: EpisodeArtifact) -> MetricValue:
        proxy_token_ids = proxy_tokens(
            episode,
            self.target_role,
            model=self.tokenizer_model,
        )
        human_token_ids = human_tokens(
            episode,
            self.target_role,
            model=self.tokenizer_model,
        )

        metadata = {
            "proxy_tokens": proxy_token_ids,
            "human_tokens": human_token_ids,
            "proxy_token_count": len(proxy_token_ids),
            "human_token_count": len(human_token_ids),
            "proxy_below_min_tokens": len(proxy_token_ids) < self.min_tokens,
            "human_below_min_tokens": (
                len(human_token_ids) < self.min_tokens if human_token_ids else True
            ),
            "tokenizer_model": self.tokenizer_model,
            "target_role": self.target_role.value,
            "params": {
                **LexicalMetricParams(
                    tokenizer_model=self.tokenizer_model,
                    target_role=self.target_role.value,
                    min_tokens=self.min_tokens,
                ).as_dict(),
                **self._params_metadata(),
            },
        }

        value = MetricValue(metric_name=self.info.name, values=[], metadata=metadata)
        episode.metric_values[self.info.name] = value
        return value

    def aggregate(self, values: Sequence[MetricValue]) -> MetricAggregate:
        proxy_scores: list[float] = []
        human_scores: list[float] = []
        episodes_missing_human = 0
        unstable_proxy = 0

        for value in values:
            proxy_tokens_list = cast(list[int], value.metadata.get("proxy_tokens", []))
            human_tokens_list = cast(list[int], value.metadata.get("human_tokens", []))

            proxy_raw = self._score(proxy_tokens_list) if proxy_tokens_list else float("nan")
            if math.isnan(proxy_raw):
                unstable_proxy += 1
                value.metadata["proxy_unstable"] = True
                proxy_raw = 0.0
            else:
                value.metadata["proxy_unstable"] = False
                proxy_scores.append(proxy_raw)
            value.metadata["proxy_raw"] = proxy_raw

            if human_tokens_list:
                human_raw = self._score(human_tokens_list)
                if math.isnan(human_raw):
                    episodes_missing_human += 1
                    value.metadata["human_raw"] = None
                    value.metadata["human_unstable"] = True
                else:
                    human_scores.append(human_raw)
                    value.metadata["human_raw"] = human_raw
                    value.metadata["human_unstable"] = False
            else:
                episodes_missing_human += 1
                value.metadata["human_raw"] = None
                value.metadata["human_unstable"] = True

        mean_human, std_human = mean_and_stdev(human_scores)

        z_scores: list[float] = []
        for value in values:
            proxy_raw_any = value.metadata.get("proxy_raw")
            if proxy_raw_any is None:
                value.values = []
                continue
            proxy_raw = float(proxy_raw_any)
            z_value = safe_z_score(proxy_raw, mean_human, std_human)
            value.values = [z_value]
            value.metadata["z_score"] = z_value
            value.metadata["baseline_mean"] = mean_human
            value.metadata["baseline_std"] = std_human
            z_scores.append(z_value)

        mean_z, stdev_z, ci_half_width = mean_stdev_ci(z_scores)

        extras = {
            "baseline_mean": mean_human,
            "baseline_std": std_human,
            "baseline_sample_size": len(human_scores),
            "mean_proxy_raw": mean(proxy_scores) if proxy_scores else 0.0,
            "mean_human_raw": mean(human_scores) if human_scores else 0.0,
            "episodes_missing_human": episodes_missing_human,
            "episodes_unstable_proxy": unstable_proxy,
            "params": values[0].metadata.get("params", {}) if values else {},
        }

        return MetricAggregate(
            metric_name=self.info.name,
            mean=mean_z,
            standard_deviation=stdev_z,
            confidence_interval=ci_half_width,
            sample_size=len(z_scores),
            extras=extras,
        )


__all__ = ["BaseLexicalDiversityMetric", "LexicalMetricParams"]
