"""Critique-then-Revise judge metric for robust realism scoring."""

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


class CritiqueThenReviseResponse(BaseModel):
    """Schema for the verdict phase of critique-then-revise."""

    score: float = Field(..., ge=0.0, le=1.0)
    explanation: str | None = Field(default=None)
    critique: str | None = Field(default=None)

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


METRIC_NAME = "metric:judge/critique_then_revise"

CTR_INFO = MetricInfo(
    name=METRIC_NAME,
    needs_references=False,
    needs_judge=True,
    supported_tasks=set(),
    category="human_likeness",
)


@register_metric(name=METRIC_NAME, metadata=CTR_INFO)
class CritiqueThenReviseMetric(JudgeMetricMixin, BaseMetric):
    """Two-pass critique followed by a final realism score."""

    info: ClassVar[MetricInfo] = CTR_INFO

    def __init__(  # noqa: PLR0913
        self,
        *,
        judge_client_name: str = "client:langchain/chat",
        judge_params: dict[str, Any] | None = None,
        bootstrap: dict[str, Any] | bool | None = None,
        prompt_version: str = prompts.CTR_PROMPT_VERSION,
        num_judge_samples: int = 3,
        max_retries: int = 1,
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
        if max_retries < 0:
            msg = "max_retries must be >= 0"
            raise ValueError(msg)
        self.prompt_version = prompt_version
        self.num_judge_samples = num_judge_samples
        self.max_retries = max_retries

    def _critique(
        self,
        conversation: str,
        few_shot_examples: list[dict[str, str]] | None = None,
    ) -> tuple[str, Any]:
        few_shot_text = prompts.format_few_shot_examples(few_shot_examples)
        messages = [
            Message(role=Role.SYSTEM, content=prompts.CTR_PROMPT_SYSTEM),
            Message(
                role=Role.USER,
                content=prompts.CTR_CRITIQUE_PROMPT_USER.format(
                    few_shot_examples=few_shot_text,
                    conversation=conversation,
                ),
            ),
        ]
        response, telemetry = self._invoke_judge_with_telemetry(messages)
        content = str(response.message.content)
        return content.strip(), telemetry

    def _verdict(
        self,
        conversation: str,
        critique: str,
        few_shot_examples: list[dict[str, str]] | None = None,
    ) -> tuple[CritiqueThenReviseResponse, Any]:
        few_shot_text = prompts.format_few_shot_examples(few_shot_examples)
        messages = [
            Message(role=Role.SYSTEM, content=prompts.CTR_PROMPT_SYSTEM),
            Message(
                role=Role.USER,
                content=prompts.CTR_VERDICT_PROMPT_USER.format(
                    few_shot_examples=few_shot_text,
                    conversation=conversation,
                    critique=critique,
                ),
            ),
        ]
        response, telemetry = self._invoke_judge_with_telemetry(messages)
        response_text = str(response.message.content)
        payload = parse_json(
            response_text,
            validate_pydantic_object=CritiqueThenReviseResponse,
        )
        return CritiqueThenReviseResponse.model_validate(payload), telemetry

    def _run_samples(
        self,
        conversation: str,
        few_shot_examples: list[dict[str, str]] | None = None,
    ) -> tuple[list[float], list[str | None], list[str], list[Any]]:
        scores: list[float] = []
        explanations: list[str | None] = []
        critiques: list[str] = []
        telemetries: list[Any] = []

        for _ in range(self.num_judge_samples):
            critique_text: str | None = None
            critique_telemetry: Any = None
            attempts = 0
            while critique_text is None and attempts <= self.max_retries:
                attempts += 1
                try:
                    critique_text, critique_telemetry = self._critique(
                        conversation, few_shot_examples
                    )
                except Exception:  # pragma: no cover - judge failure path
                    if attempts > self.max_retries:
                        raise
                    critique_text = None
                    critique_telemetry = None
            if critique_text is None:  # pragma: no cover - judge failure path
                msg = "Failed to get critique from judge"
                raise RuntimeError(msg)
            verdict, verdict_telemetry = self._verdict(
                conversation, critique_text, few_shot_examples
            )
            scores.append(verdict.score)
            explanations.append(verdict.explanation)
            critique_output = verdict.critique if verdict.critique is not None else critique_text
            critiques.append(critique_output or "")
            # Collect both critique and verdict telemetry (2 judge calls per sample)
            telemetries.append({"critique": critique_telemetry, "verdict": verdict_telemetry})

        return scores, explanations, critiques, telemetries

    def evaluate(self, episode: EpisodeArtifact) -> MetricValue:
        """Run critique-then-revise flow to score realism."""

        proxy_conversation = episode.turns
        if not proxy_conversation:
            msg = f"Episode {episode.spec.episode_id} has no proxy conversation"
            raise ValueError(msg)

        # Extract few-shot examples from metadata
        few_shot_examples = (episode.spec.metadata or {}).get("few_shot_user_examples")

        conversation = self._format_conversation(proxy_conversation)

        scores, explanations, critiques, telemetries = self._run_samples(
            conversation, few_shot_examples
        )

        for score, explanation, critique, telemetry_pair in zip(
            scores, explanations, critiques, telemetries, strict=False
        ):
            # Use verdict telemetry as primary (the final judgment)
            verdict_telemetry = telemetry_pair.get("verdict")
            judge_verdict = JudgeVerdict(
                score=score,
                label="ctr",
                reason=explanation,
                raw={
                    "score": score,
                    "explanation": explanation,
                    "critique": critique,
                },
                telemetry=verdict_telemetry,
            )
            episode.judge_verdicts.append(judge_verdict)

        mean_score = mean(scores)

        metadata: dict[str, Any] = {
            "prompt_version": self.prompt_version,
            "judge_client": self.judge_client_name,
            "scores": scores,
            "explanations": explanations,
            "critiques": critiques,
        }

        controls: dict[str, Any] = {}
        if self.compute_controls:
            real_conversation = resolve_reference_conversation(episode)
            if real_conversation:
                real_conv_str = self._format_conversation(real_conversation)
                hh_scores, hh_explanations, hh_critiques, hh_telemetries = self._run_samples(
                    real_conv_str, few_shot_examples
                )
                controls["hh"] = {
                    "scores": hh_scores,
                    "explanations": hh_explanations,
                    "critiques": hh_critiques,
                    "mean": mean(hh_scores),
                }
                for score, explanation, critique, telemetry_pair in zip(
                    hh_scores, hh_explanations, hh_critiques, hh_telemetries, strict=False
                ):
                    verdict_telemetry = telemetry_pair.get("verdict")
                    episode.judge_verdicts.append(
                        JudgeVerdict(
                            score=score,
                            label="ctr_hh",
                            reason=explanation,
                            raw={
                                "score": score,
                                "explanation": explanation,
                                "critique": critique,
                            },
                            telemetry=verdict_telemetry,
                        )
                    )

        if controls:
            metadata["controls"] = controls

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

        control_means: list[float] = []
        for value in values:
            controls = value.metadata.get("controls")
            if isinstance(controls, dict):
                hh = controls.get("hh")
                if isinstance(hh, dict) and "mean" in hh:
                    control_means.append(float(hh["mean"]))

        if control_means:
            aggregate.extras["reference_mean"] = sum(control_means) / len(control_means)
            aggregate.extras["reference_count"] = len(control_means)
        else:
            aggregate.extras["reference_mean"] = None
            aggregate.extras["reference_count"] = 0

        return aggregate


__all__ = [
    "CTR_INFO",
    "CritiqueThenReviseMetric",
]
