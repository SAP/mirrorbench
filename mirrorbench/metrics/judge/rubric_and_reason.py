"""Rubric-and-Reason judge metric (G-Eval style realism score)."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean
from typing import Any, ClassVar

from pydantic import BaseModel, Field, model_validator

from mirrorbench.core.models.episodes import EpisodeArtifact
from mirrorbench.core.models.messages import JudgeVerdict, Message, Role
from mirrorbench.core.models.registry import MetricInfo
from mirrorbench.core.models.run import MetricAggregate, MetricValue
from mirrorbench.core.registry import BaseMetric
from mirrorbench.core.registry.decorators import register_metric
from mirrorbench.metrics.judge import prompts
from mirrorbench.metrics.judge.common import JudgeMetricMixin
from mirrorbench.metrics.util import parse_json, resolve_reference_conversation


class RubricAndReasonResponse(BaseModel):
    """Schema for parsing rubric-and-reason responses."""

    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _convert_verdict_to_score(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        verdict = data.get("verdict", None)
        if isinstance(verdict, str):
            verdict = verdict.strip().upper()
        if not isinstance(verdict, str) or verdict not in {"YES", "NO"}:
            raise ValueError('verdict must be "YES" or "NO"')
        score = 1.0 if verdict == "YES" else 0.0
        data["score"] = score
        return data


METRIC_NAME = "metric:judge/rubric_and_reason"

RNR_INFO = MetricInfo(
    name=METRIC_NAME,
    needs_references=False,
    needs_judge=True,
    supported_tasks=set(),
    category="human_likeness",
)


@register_metric(name=METRIC_NAME, metadata=RNR_INFO)
class RubricAndReasonMetric(JudgeMetricMixin, BaseMetric):
    """Absolute realism score using a rubric-driven judge prompt."""

    info: ClassVar[MetricInfo] = RNR_INFO

    def __init__(  # noqa: PLR0913
        self,
        *,
        judge_client_name: str = "client:langchain/chat",
        judge_params: dict[str, Any] | None = None,
        bootstrap: dict[str, Any] | bool | None = None,
        prompt_version: str = prompts.RNR_PROMPT_VERSION,
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
        self.prompt_version = prompt_version
        self.num_judge_samples = num_judge_samples

    def _score_conversation(
        self,
        conversation: str,
        few_shot_examples: list[dict[str, str]] | None = None,
    ) -> JudgeVerdict:
        few_shot_text = prompts.format_few_shot_examples(few_shot_examples)
        messages = [
            Message(role=Role.SYSTEM, content=prompts.RNR_PROMPT_SYSTEM),
            Message(
                role=Role.USER,
                content=prompts.RNR_PROMPT_USER.format(
                    few_shot_examples=few_shot_text,
                    conversation=conversation,
                ),
            ),
        ]
        response, telemetry = self._invoke_judge_with_telemetry(messages)
        response_text = response.message.content
        payload = parse_json(response_text, validate_pydantic_object=RubricAndReasonResponse)
        parsed = RubricAndReasonResponse.model_validate(payload)
        return JudgeVerdict(
            score=parsed.score,
            label="rnr",
            reason=parsed.reasoning,
            raw=payload,
            telemetry=telemetry,
        )

    def _run_samples(
        self,
        conversation: str,
        few_shot_examples: list[dict[str, str]] | None = None,
    ) -> list[JudgeVerdict]:
        verdicts: list[JudgeVerdict] = []
        for _ in range(self.num_judge_samples):
            verdicts.append(self._score_conversation(conversation, few_shot_examples))
        return verdicts

    def evaluate(self, episode: EpisodeArtifact) -> MetricValue:
        """Score how human-like the proxy conversation sounds."""

        proxy_conversation = episode.turns
        if not proxy_conversation:
            msg = f"Episode {episode.spec.episode_id} has no proxy conversation"
            raise ValueError(msg)

        # Extract few-shot examples from metadata
        few_shot_examples = (episode.spec.metadata or {}).get("few_shot_user_examples")

        conversation = self._format_conversation(proxy_conversation)
        verdicts = self._run_samples(conversation, few_shot_examples)
        for verdict in verdicts:
            episode.judge_verdicts.append(verdict)

        proxy_scores = [verdict.score for verdict in verdicts]
        proxy_reasonings = [verdict.reason for verdict in verdicts]

        metadata: dict[str, Any] = {
            "prompt_version": self.prompt_version,
            "judge_client": self.judge_client_name,
            "scores": proxy_scores,
            "reasonings": proxy_reasonings,
            "reasoning": proxy_reasonings[0] if proxy_reasonings else None,
        }

        controls: dict[str, dict[str, Any]] = {}
        if self.compute_controls:
            real_conversation = resolve_reference_conversation(episode)
            if real_conversation:
                real_conv_str = self._format_conversation(real_conversation)
                real_verdicts = self._run_samples(real_conv_str, few_shot_examples)
                real_scores = [verdict.score for verdict in real_verdicts]
                controls["hh"] = {
                    "scores": real_scores,
                    "reasonings": [verdict.reason for verdict in real_verdicts],
                    "score": mean(real_scores),
                    "reasoning": real_verdicts[0].reason if real_verdicts else None,
                }
                for verdict in real_verdicts:
                    episode.judge_verdicts.append(
                        JudgeVerdict(
                            score=verdict.score,
                            label="rnr_hh",
                            reason=verdict.reason,
                            raw={
                                "score": verdict.score,
                                "reasoning": verdict.reason,
                            },
                        )
                    )

        if controls:
            metadata["controls"] = controls

        mean_score = mean(proxy_scores) if proxy_scores else 0.0
        metadata["score"] = mean_score
        metric_value = MetricValue(
            metric_name=self.info.name,
            values=[mean_score],
            metadata=metadata,
        )
        episode.metric_values[self.info.name] = metric_value
        return metric_value

    def aggregate(self, values: Sequence[MetricValue]) -> MetricAggregate:
        extras = {
            "prompt_version": self.prompt_version,
            "judge_client": self.judge_client_name,
        }
        aggregate = self._aggregate_scalar_metric(values, extras=extras)

        control_scores: list[float] = []
        for value in values:
            controls = value.metadata.get("controls")
            if isinstance(controls, dict):
                hh = controls.get("hh")
                if isinstance(hh, dict) and "score" in hh:
                    control_scores.append(float(hh["score"]))

        if control_scores:
            aggregate.extras["reference_mean"] = mean(control_scores)
            aggregate.extras["reference_count"] = len(control_scores)
        else:
            aggregate.extras["reference_mean"] = None
            aggregate.extras["reference_count"] = 0

        return aggregate


__all__ = [
    "RNR_INFO",
    "RubricAndReasonMetric",
]
