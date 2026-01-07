"""GTEval metric for conversation-level comparison using LLM-as-a-judge."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean
from typing import Any, ClassVar

from pydantic import BaseModel, Field

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


class GTEvalJudgeResponse(BaseModel):
    """Pydantic model for parsing GTEval judge responses."""

    reasoning: str = Field(..., description="Detailed explanation of the evaluation")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score between 0.0 and 1.0")


METRIC_NAME = "metric:judge/gteval"

GTEVAL_METRIC_INFO = MetricInfo(
    name=METRIC_NAME,
    needs_references=True,
    needs_judge=True,
    supported_tasks=set(),
    category="human_likeness",
)


@register_metric(name=METRIC_NAME, metadata=GTEVAL_METRIC_INFO)
class GTEvalMetric(JudgeMetricMixin, BaseMetric):
    """GTEval judge-based metric for comparing proxy vs. real user conversations."""

    info: ClassVar[MetricInfo] = GTEVAL_METRIC_INFO

    def __init__(  # noqa: PLR0913
        self,
        *,
        judge_client_name: str = "client:langchain/chat",
        judge_params: dict[str, Any] | None = None,
        rubric_version: str = "1.0",
        bootstrap: dict[str, Any] | bool | None = None,
        compute_controls: bool = True,
        num_judge_samples: int = 3,
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
        self.rubric_version = rubric_version
        self.num_judge_samples = num_judge_samples

    def _parse_judge_response(self, response_text: str, telemetry: Any = None) -> JudgeVerdict:
        """Parse the judge's JSON response to extract score and reasoning."""

        response_data = parse_json(
            response_text,
            validate_pydantic_object=GTEvalJudgeResponse,
        )
        parsed_response = GTEvalJudgeResponse.model_validate(response_data)

        return JudgeVerdict(
            score=parsed_response.score,
            label="gteval",
            confidence=None,
            reason=parsed_response.reasoning,
            raw=response_data,
            telemetry=telemetry,
        )

    def _score_pair(self, *, real_conv: str, proxy_conv: str) -> list[JudgeVerdict]:
        """Score a pair of conversations using the configured judge."""

        prompt = prompts.GTEVAL_PROMPT_TEMPLATE.format(
            real_conversation=real_conv,
            proxy_conversation=proxy_conv,
        )
        verdicts: list[JudgeVerdict] = []
        for _ in range(self.num_judge_samples):
            response, telemetry = self._invoke_judge_with_telemetry(
                [Message(role=Role.USER, content=prompt)]
            )
            verdicts.append(self._parse_judge_response(response.message.content, telemetry))
        return verdicts

    def evaluate(self, episode: EpisodeArtifact) -> MetricValue:
        """Evaluate conversation-level similarity between proxy and real user responses."""

        real_conversation = resolve_reference_conversation(episode)
        proxy_conversation = episode.turns

        if not proxy_conversation:
            msg = f"Episode {episode.spec.episode_id} has no proxy conversation"
            raise ValueError(msg)
        if not real_conversation:
            msg = f"Episode {episode.spec.episode_id} has no real conversation"
            raise ValueError(msg)

        real_conv_str = self._format_conversation(real_conversation)
        proxy_conv_str = self._format_conversation(proxy_conversation)

        verdicts = self._score_pair(real_conv=real_conv_str, proxy_conv=proxy_conv_str)
        for verdict in verdicts:
            episode.judge_verdicts.append(verdict)

        proxy_user_turns = [msg for msg in proxy_conversation if msg.role == Role.USER]
        real_user_turns = [msg for msg in real_conversation if msg.role == Role.USER]

        metadata: dict[str, Any] = {
            "rubric_version": self.rubric_version,
            "judge_client": self.judge_client_name,
            "prompt_version": prompts.GTEVAL_PROMPT_VERSION,
            "proxy_user_turn_count": len(proxy_user_turns),
            "real_user_turn_count": len(real_user_turns),
            "scores": [verdict.score for verdict in verdicts],
            "reasonings": [verdict.reason for verdict in verdicts],
            "reasoning": verdicts[0].reason if verdicts else None,
        }

        controls: dict[str, dict[str, Any]] = {}
        if self.compute_controls:
            if real_conversation:
                hh_verdicts = self._score_pair(
                    real_conv=real_conv_str,
                    proxy_conv=real_conv_str,
                )
                hh_scores = [verdict.score for verdict in hh_verdicts]
                controls["hh"] = {
                    "scores": hh_scores,
                    "reasonings": [verdict.reason for verdict in hh_verdicts],
                    "score": mean(hh_scores),
                    "reasoning": hh_verdicts[0].reason if hh_verdicts else None,
                }
                for verdict in hh_verdicts:
                    episode.judge_verdicts.append(
                        JudgeVerdict(
                            score=verdict.score,
                            label="gteval_hh",
                            reason=verdict.reason,
                            raw={"score": verdict.score, "reason": verdict.reason},
                            telemetry=verdict.telemetry,
                        )
                    )
            if proxy_conversation:
                pp_verdicts = self._score_pair(
                    real_conv=proxy_conv_str,
                    proxy_conv=proxy_conv_str,
                )
                pp_scores = [verdict.score for verdict in pp_verdicts]
                controls["pp"] = {
                    "scores": pp_scores,
                    "reasonings": [verdict.reason for verdict in pp_verdicts],
                    "score": mean(pp_scores),
                    "reasoning": pp_verdicts[0].reason if pp_verdicts else None,
                }
                for verdict in pp_verdicts:
                    episode.judge_verdicts.append(
                        JudgeVerdict(
                            score=verdict.score,
                            label="gteval_pp",
                            reason=verdict.reason,
                            raw={"score": verdict.score, "reason": verdict.reason},
                            telemetry=verdict.telemetry,
                        )
                    )

        if controls:
            metadata["controls"] = controls

        control_label = (episode.spec.metadata or {}).get("judge_control")
        if isinstance(control_label, str) and control_label.upper() in {HH_LABEL, PP_LABEL}:
            metadata[CONTROL_METADATA_KEY] = control_label.upper()

        main_scores = metadata.get("scores", [])
        value_mean = mean(main_scores) if main_scores else 0.0
        metadata["score"] = value_mean
        metric_value = MetricValue(
            metric_name=self.info.name,
            values=[value_mean],
            metadata=metadata,
        )
        episode.metric_values[self.info.name] = metric_value
        return metric_value

    def aggregate(self, values: Sequence[MetricValue]) -> MetricAggregate:
        """Aggregate per-episode GTEval scores across the full evaluation run."""

        control_labels = {HH_LABEL, PP_LABEL}
        eval_values = [
            value
            for value in values
            if value.metadata.get(CONTROL_METADATA_KEY) not in control_labels
        ]

        total_proxy_turns = sum(
            value.metadata.get("proxy_user_turn_count", 0) for value in eval_values
        )
        total_real_turns = sum(
            value.metadata.get("real_user_turn_count", 0) for value in eval_values
        )

        extras = {
            "rubric_version": self.rubric_version,
            "judge_client": self.judge_client_name,
            "prompt_version": prompts.GTEVAL_PROMPT_VERSION,
            "total_proxy_user_turns": total_proxy_turns,
            "total_real_user_turns": total_real_turns,
        }

        aggregate = self._aggregate_scalar_metric(eval_values, extras=extras)

        raw_scores = [score for value in eval_values for score in value.values]
        anchors = derive_anchors(values)
        calibrated_scores = (
            apply_linear_calibration(raw_scores, anchors)
            if anchors is not None and raw_scores
            else None
        )
        aggregate.extras["calibration"] = calibration_summary(
            raw_mean=aggregate.mean,
            anchors=anchors,
            calibrated_scores=calibrated_scores,
        )

        return aggregate


__all__ = [
    "GTEVAL_METRIC_INFO",
    "GTEvalMetric",
]
