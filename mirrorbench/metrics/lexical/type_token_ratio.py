"""Type-Token Ratio (TTR) metric for lexical diversity evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from mirrorbench.core.models.episodes import EpisodeArtifact
from mirrorbench.core.models.messages import Role
from mirrorbench.core.models.registry import MetricInfo
from mirrorbench.core.models.run import MetricAggregate, MetricValue
from mirrorbench.core.registry import BaseMetric
from mirrorbench.core.registry.decorators import register_metric
from mirrorbench.metrics.util.text import tokenize

METRIC_NAME = "metric:lexical/ttr"

TTR_METRIC_INFO = MetricInfo(
    name=METRIC_NAME,
    needs_references=False,
    needs_judge=False,
    supported_tasks=set(),  # Universal metric, works with all tasks
    category="lexical_diversity",
)


@register_metric(name=METRIC_NAME, metadata=TTR_METRIC_INFO)
class TypeTokenRatioMetric(BaseMetric):
    """Compute Type-Token Ratio (TTR) to measure lexical diversity.

    TTR is the ratio of unique token types to total tokens, capturing
    vocabulary richness without requiring reference data. This implementation
    operates at corpus scope: per-episode evaluation tokenizes and stores all
    token IDs, and aggregation combines all tokens from all episodes to compute
    a true corpus-level TTR based on the ratio of unique tokens to total tokens.

    By default, the metric analyzes USER role messages to assess the diversity
    of user utterances, but can be configured to analyze ASSISTANT responses
    or any other role via the target_role parameter.

    Implementation:
        - evaluate(): Tokenizes text and stores token IDs in metadata
        - aggregate(): Combines all token IDs from all episodes, counts unique tokens,
          and computes TTR = unique_tokens / total_tokens

    References:
        Rajeshwari Depala. "Empirical Laws: Type-Token Ratio (TTR)."
        https://medium.com/@rajeshwaridepala/empirical-laws-type-token-ratio-ttr-e8b247174b85

    Attributes:
        info: Metric metadata for registry integration.
        min_tokens: Minimum token count threshold to avoid instability on very short texts.
        tokenizer_model: Model name for tiktoken tokenizer.
        target_role: The message role to analyze for lexical diversity.
    """

    info: ClassVar[MetricInfo] = TTR_METRIC_INFO

    def __init__(
        self,
        *,
        min_tokens: int = 5,
        tokenizer_model: str = "gpt-4o",
        target_role: Role | str = Role.USER,
    ) -> None:
        """Initialize the TTR metric.

        Args:
            min_tokens: Minimum number of tokens required for stable TTR calculation.
                Episodes with fewer tokens will still contribute counts but will be
                flagged in metadata.
            tokenizer_model: Model name for tiktoken tokenizer (default: "gpt-4o").
                Common options: "gpt-4o", "gpt-4", "gpt-3.5-turbo".
            target_role: The role to analyze for lexical diversity (default: Role.USER).
                Set to Role.ASSISTANT to analyze assistant responses instead.
        """
        if isinstance(target_role, str):
            try:
                target_role = Role(target_role.lower())
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    "target_role must be a valid message role (system|user|assistant|tool|judge)"
                ) from exc

        self.min_tokens = min_tokens
        self.tokenizer_model = tokenizer_model
        self.target_role = target_role

    def evaluate(self, episode: EpisodeArtifact) -> MetricValue:
        """Extract token IDs from messages with the target role.

        This method tokenizes all target role messages and stores the token IDs
        in metadata. The actual TTR computation happens in aggregate() by combining
        all tokens from all episodes.

        Args:
            episode: The completed episode artifact containing proxy-generated turns.

        Returns:
            MetricValue with empty values list and metadata containing:
                - token_ids: List of all token IDs from target role messages
                - token_count: Total number of tokens
                - min_tokens: Minimum token threshold used
                - below_threshold: True if token count is below min_tokens
                - target_role: The role that was analyzed
        """
        # Extract messages with the target role from turns
        target_turns = [msg for msg in episode.turns if msg.role == self.target_role]

        # Concatenate all target role turn content
        all_text = " ".join(msg.content for msg in target_turns)

        # Tokenize using tiktoken for the specified model
        token_ids = tokenize(all_text, model=self.tokenizer_model)

        token_count = len(token_ids)
        below_threshold = token_count < self.min_tokens

        metadata = {
            "token_ids": token_ids,  # Store actual token IDs for corpus-level aggregation
            "token_count": token_count,
            "min_tokens": self.min_tokens,
            "below_threshold": below_threshold,
            "tokenizer_model": self.tokenizer_model,
            "target_role": self.target_role.value,
        }

        metric_value = MetricValue(
            metric_name=self.info.name,
            values=[],  # No standalone episode score
            metadata=metadata,
        )

        episode.metric_values[self.info.name] = metric_value

        return metric_value

    def aggregate(self, values: Sequence[MetricValue]) -> MetricAggregate:
        """Combine all tokens from all episodes and compute corpus-level TTR.

        This method collects all token IDs from all episodes, combines them into
        a single corpus, and computes the true Type-Token Ratio as the ratio of
        unique token IDs to total token count.

        Args:
            values: Sequence of per-episode metric values containing token_ids.

        Returns:
            MetricAggregate with:
                - mean: Corpus-level TTR (unique_tokens / total_tokens)
                - sample_size: Number of episodes processed
                - extras: Dictionary containing total_tokens, total_types,
                    episodes_below_threshold
        """
        all_token_ids: list[int] = []
        episodes_below_threshold = 0

        # Collect all token IDs from all episodes
        for value in values:
            metadata = value.metadata
            token_ids = metadata.get("token_ids", [])
            below_threshold = metadata.get("below_threshold", False)

            all_token_ids.extend(token_ids)
            if below_threshold:
                episodes_below_threshold += 1

        # Compute true corpus-level TTR
        total_tokens = len(all_token_ids)
        unique_tokens = len(set(all_token_ids))
        ttr = 0.0 if total_tokens == 0 else unique_tokens / total_tokens

        return MetricAggregate(
            metric_name=self.info.name,
            mean=ttr,
            standard_deviation=0.0,  # Not applicable for corpus-level metric, thus 0.0 always
            sample_size=len(values),
            extras={
                "total_tokens": total_tokens,
                "total_types": unique_tokens,
                "episodes_below_threshold": episodes_below_threshold,
                "min_tokens_threshold": self.min_tokens,
                "tokenizer_model": self.tokenizer_model,
                "target_role": self.target_role.value,
            },
        )


__all__ = [
    "TTR_METRIC_INFO",
    "TypeTokenRatioMetric",
]
