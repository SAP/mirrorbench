"""Pairwise indistinguishability metric using an LLM judge."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from statistics import mean
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, model_validator

from mirrorbench.core.models.episodes import EpisodeArtifact
from mirrorbench.core.models.messages import JudgeVerdict, Message, Role
from mirrorbench.core.models.registry import MetricInfo
from mirrorbench.core.models.run import MetricAggregate, MetricValue
from mirrorbench.core.registry import BaseMetric
from mirrorbench.core.registry.decorators import register_metric
from mirrorbench.metrics.judge import prompts
from mirrorbench.metrics.judge.calibration import (
    CONTROL_METADATA_KEY,
    HH_LABEL,
    PP_LABEL,
    apply_linear_calibration,
    calibration_summary,
    derive_anchors,
)
from mirrorbench.metrics.judge.common import JudgeMetricMixin
from mirrorbench.metrics.util import parse_json, resolve_reference_conversation


class PairwiseJudgeResponse(BaseModel):
    """Schema for the pairwise judge response."""

    verdict: Literal["A", "B", "TIE"] = Field(..., description="A, B, or Tie")
    reasoning: str | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _check_verdict(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data["verdict"] = str(data.get("verdict", "")).strip().upper()
        return data


COIN_FLIP_THRESHOLD = 0.5
PARITY_NEUTRAL = 0.5


METRIC_NAME = "metric:judge/pi_pairwise"

PAIRWISE_INFO = MetricInfo(
    name=METRIC_NAME,
    needs_references=True,
    needs_judge=True,
    supported_tasks=set(),
    category="human_likeness",
)


@register_metric(name=METRIC_NAME, metadata=PAIRWISE_INFO)
class PairwiseIndistinguishabilityMetric(JudgeMetricMixin, BaseMetric):
    """Arena-style human vs. proxy comparison using an LLM judge."""

    info: ClassVar[MetricInfo] = PAIRWISE_INFO

    def __init__(  # noqa: PLR0913
        self,
        *,
        judge_client_name: str = "client:langchain/chat",
        judge_params: dict[str, Any] | None = None,
        num_judge_samples: int = 3,
        bootstrap: dict[str, Any] | bool | None = None,
        prompt_version: str = prompts.PAIRWISE_PROMPT_VERSION,
        compute_controls: bool = True,
    ) -> None:
        super().__init__(
            judge_client_name=judge_client_name,
            judge_params=judge_params,
            bootstrap=bootstrap,
            compute_controls=compute_controls,
        )
        if num_judge_samples <= 0:
            msg = "num_judge_samples must be >= 1"
            raise ValueError(msg)
        self.num_judge_samples = num_judge_samples
        self.prompt_version = prompt_version

    def _build_messages(
        self,
        conversation_a: str,
        conversation_b: str,
        few_shot_examples: list[dict[str, str]] | None = None,
    ) -> list[Message]:
        """Compose the chat messages sent to the judge model."""

        few_shot_text = prompts.format_few_shot_examples(few_shot_examples)
        return [
            Message(role=Role.SYSTEM, content=prompts.PAIRWISE_PROMPT_SYSTEM),
            Message(
                role=Role.USER,
                content=prompts.PAIRWISE_PROMPT_USER.format(
                    few_shot_examples=few_shot_text,
                    conversation_a=conversation_a,
                    conversation_b=conversation_b,
                ),
            ),
        ]

    def _parse_response(self, response_text: str) -> PairwiseJudgeResponse:
        payload = parse_json(response_text, validate_pydantic_object=PairwiseJudgeResponse)
        payload["verdict"] = str(payload["verdict"]).strip().upper()
        return PairwiseJudgeResponse.model_validate(payload)

    def _run_samples(
        self,
        *,
        human_conv: str,
        proxy_conv: str,
        episode_id: str,
        base_seed: int,
        few_shot_examples: list[dict[str, str]] | None = None,
    ) -> tuple[list[float], list[str], list[str | None], list[str], list[Any]]:
        wins: list[float] = []
        verdicts: list[str] = []
        reasonings: list[str | None] = []
        orders: list[str] = []
        telemetries: list[Any] = []

        for sample_idx in range(self.num_judge_samples):
            seed = self._rng_seed(
                episode_id=episode_id,
                base_seed=base_seed + sample_idx,
                salt=f"pairwise-control-{self.prompt_version}",
            )
            rng = random.Random(seed)
            proxy_first = rng.random() < COIN_FLIP_THRESHOLD

            if proxy_first:
                conversation_a = proxy_conv
                conversation_b = human_conv
                proxy_slot = "A"
                order = "PH"
            else:
                conversation_a = human_conv
                conversation_b = proxy_conv
                proxy_slot = "B"
                order = "HP"

            messages = self._build_messages(conversation_a, conversation_b, few_shot_examples)
            response, telemetry = self._invoke_judge_with_telemetry(messages)
            parsed = self._parse_response(response.message.content)

            verdict = parsed.verdict.strip().upper()
            if verdict not in {"A", "B", "TIE"}:
                msg = f"Unexpected verdict '{parsed.verdict}'"
                raise ValueError(msg)

            if verdict == proxy_slot:
                win_value = 1.0
            elif verdict == "TIE":
                win_value = 0.5
            else:
                win_value = 0.0

            wins.append(win_value)
            verdicts.append(verdict)
            reasonings.append(parsed.reasoning)
            orders.append(order)
            telemetries.append(telemetry)

        return wins, verdicts, reasonings, orders, telemetries

    def evaluate(self, episode: EpisodeArtifact) -> MetricValue:
        """Run pairwise comparison between real and proxy conversations."""

        real_conversation = resolve_reference_conversation(episode)
        proxy_conversation = episode.turns

        if not proxy_conversation:
            msg = f"Episode {episode.spec.episode_id} missing proxy conversation"
            raise ValueError(msg)
        if not real_conversation:
            msg = f"Episode {episode.spec.episode_id} missing reference conversation"
            raise ValueError(msg)

        real_str = self._format_conversation(real_conversation)
        proxy_str = self._format_conversation(proxy_conversation)

        # Extract few-shot examples from metadata
        few_shot_examples = (episode.spec.metadata or {}).get("few_shot_user_examples")

        base_seed = int((episode.spec.metadata or {}).get("seed", 0))
        wins, verdicts, reasonings, orders, telemetries = self._run_samples(
            human_conv=real_str,
            proxy_conv=proxy_str,
            episode_id=episode.spec.episode_id,
            base_seed=base_seed,
            few_shot_examples=few_shot_examples,
        )

        for verdict, reasoning, order, win_value, telemetry in zip(
            verdicts, reasonings, orders, wins, telemetries, strict=False
        ):
            proxy_slot = "A" if order == "PH" else "B"
            episode.judge_verdicts.append(
                JudgeVerdict(
                    score=win_value,
                    label=verdict,
                    reason=reasoning,
                    raw={
                        "verdict": verdict,
                        "reasoning": reasoning,
                        "order": order,
                        "proxy_slot": proxy_slot,
                    },
                    telemetry=telemetry,
                )
            )

        mean_win = mean(wins)
        tie_count = sum(1 for verdict in verdicts if verdict == "TIE")

        metadata: dict[str, Any] = {
            "judge_client": self.judge_client_name,
            "prompt_version": self.prompt_version,
            "verdicts": verdicts,
            "reasonings": reasonings,
            "orders": orders,
            "tie_count": tie_count,
            "sample_count": self.num_judge_samples,
        }

        controls: dict[str, dict[str, Any]] = {}
        if self.compute_controls:
            if real_conversation:
                hh_wins, hh_verdicts, hh_reasonings, hh_orders, hh_telemetries = self._run_samples(
                    human_conv=real_str,
                    proxy_conv=real_str,
                    episode_id=f"{episode.spec.episode_id}:hh",
                    base_seed=base_seed,
                    few_shot_examples=few_shot_examples,
                )
                controls["hh"] = {
                    "score": mean(hh_wins),
                    "verdicts": hh_verdicts,
                    "reasonings": hh_reasonings,
                    "orders": hh_orders,
                }
                # Add control verdicts to episode
                for verdict, reasoning, order, win_value, telemetry in zip(
                    hh_verdicts, hh_reasonings, hh_orders, hh_wins, hh_telemetries, strict=False
                ):
                    episode.judge_verdicts.append(
                        JudgeVerdict(
                            score=win_value,
                            label=f"{verdict}_hh",
                            reason=reasoning,
                            raw={"verdict": verdict, "reasoning": reasoning, "order": order},
                            telemetry=telemetry,
                        )
                    )
            if proxy_conversation:
                pp_wins, pp_verdicts, pp_reasonings, pp_orders, pp_telemetries = self._run_samples(
                    human_conv=proxy_str,
                    proxy_conv=proxy_str,
                    episode_id=f"{episode.spec.episode_id}:pp",
                    base_seed=base_seed,
                    few_shot_examples=few_shot_examples,
                )
                controls["pp"] = {
                    "score": mean(pp_wins),
                    "verdicts": pp_verdicts,
                    "reasonings": pp_reasonings,
                    "orders": pp_orders,
                }
                # Add control verdicts to episode
                for verdict, reasoning, order, win_value, telemetry in zip(
                    pp_verdicts, pp_reasonings, pp_orders, pp_wins, pp_telemetries, strict=False
                ):
                    episode.judge_verdicts.append(
                        JudgeVerdict(
                            score=win_value,
                            label=f"{verdict}_pp",
                            reason=reasoning,
                            raw={"verdict": verdict, "reasoning": reasoning, "order": order},
                            telemetry=telemetry,
                        )
                    )

        if controls:
            metadata["controls"] = controls

        control_label = (episode.spec.metadata or {}).get("judge_control")
        if isinstance(control_label, str) and control_label.upper() in {HH_LABEL, PP_LABEL}:
            metadata[CONTROL_METADATA_KEY] = control_label.upper()

        metric_value = MetricValue(
            metric_name=self.info.name,
            values=[mean_win],
            metadata=metadata,
        )
        episode.metric_values[self.info.name] = metric_value
        return metric_value

    def aggregate(self, values: Sequence[MetricValue]) -> MetricAggregate:
        """Aggregate win-rates and report calibration/extras."""

        control_labels = {HH_LABEL, PP_LABEL}
        eval_values = [
            value
            for value in values
            if value.metadata.get(CONTROL_METADATA_KEY) not in control_labels
        ]
        raw_scores = [score for value in eval_values for score in value.values]
        total_ties = sum(value.metadata.get("tie_count", 0) for value in eval_values)
        total_samples = sum(value.metadata.get("sample_count", 0) for value in eval_values)

        order_counter: Counter[str] = Counter()
        for value in eval_values:
            order_counter.update(value.metadata.get("orders", []))

        control_counter: Counter[str] = Counter(
            label
            for value in values
            for label in [value.metadata.get(CONTROL_METADATA_KEY)]
            if isinstance(label, str) and label in control_labels
        )

        extras = {
            "prompt_version": self.prompt_version,
            "judge_client": self.judge_client_name,
            "tie_count": total_ties,
            "sample_count": total_samples,
            "order_counts": dict(order_counter),
            "control_counts": {label: control_counter.get(label, 0) for label in control_labels},
        }

        aggregate = self._aggregate_scalar_metric(eval_values, extras=extras)
        mean_score = aggregate.mean
        anchors = derive_anchors(values)
        calibrated_scores = (
            apply_linear_calibration(raw_scores, anchors)
            if anchors is not None and raw_scores
            else None
        )
        aggregate.extras["parity_gap"] = mean_score - PARITY_NEUTRAL
        aggregate.extras["tie_rate"] = (total_ties / total_samples) if total_samples else 0.0
        aggregate.extras["calibration"] = calibration_summary(
            raw_mean=mean_score,
            anchors=anchors,
            calibrated_scores=calibrated_scores,
        )
        return aggregate


__all__ = [
    "PAIRWISE_INFO",
    "PairwiseIndistinguishabilityMetric",
]
