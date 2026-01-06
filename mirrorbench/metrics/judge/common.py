"""Shared utilities for judge-based metrics."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from mirrorbench.cache import get_cache_manager
from mirrorbench.core.config import CacheConfig
from mirrorbench.core.constants import DEFAULT_CACHE_TTL_SECONDS, REGISTRY_GROUP_MODEL_CLIENTS
from mirrorbench.core.models.messages import Message, TurnTelemetry
from mirrorbench.core.models.registry import MetricInfo, ModelClientInfo
from mirrorbench.core.models.run import MetricAggregate, MetricValue
from mirrorbench.core.registry import registry
from mirrorbench.io.paths import Paths
from mirrorbench.metrics.util.bootstrap import BootstrapConfig, bootstrap_mean
from mirrorbench.metrics.util.stats import mean_stdev_ci
from mirrorbench.model_clients.caching_wrapper import CachingChatClient
from mirrorbench.model_clients.telemetry import usage_to_turn_telemetry
from mirrorbench.model_clients.utils import is_chat_client


def _resolve_bootstrap_config(
    bootstrap: dict[str, Any] | bool | None,
) -> tuple[bool, BootstrapConfig]:
    """Normalise bootstrap configuration passed to judge metrics."""

    if isinstance(bootstrap, dict):
        config = BootstrapConfig(**bootstrap)
        enabled = config.iterations > 0
        return enabled, config

    if bootstrap is True:
        return True, BootstrapConfig()

    return False, BootstrapConfig()


class JudgeMetricMixin:
    """Mixin adding judge client management and aggregation helpers."""

    info: ClassVar[MetricInfo]

    def __init__(
        self,
        *,
        judge_client_name: str,
        judge_params: dict[str, Any] | None = None,
        bootstrap: dict[str, Any] | bool | None = None,
        compute_controls: bool = True,
    ) -> None:
        self.judge_client_name = judge_client_name
        self.judge_params = judge_params or {}
        self._judge_client: Any | None = None
        self._cache_config = CacheConfig(
            enabled=True,
            ttl_seconds=DEFAULT_CACHE_TTL_SECONDS,
            backend="sqlite",
        )
        self._paths: Paths | None = None
        self._bootstrap_enabled, self.bootstrap_config = _resolve_bootstrap_config(bootstrap)
        self._cache_namespace_prefix = (
            f"judge:{getattr(self.info, 'name', 'unknown')}:{judge_client_name}"
        )
        self.compute_controls = compute_controls

    # ---------------------------------------------------------------------
    # Cache handling
    # ---------------------------------------------------------------------
    def configure_cache(self, *, cache_config: CacheConfig, paths: Paths) -> None:
        """Override the cache configuration used for judge calls."""

        self._cache_config = cache_config
        self._paths = paths

    # ---------------------------------------------------------------------
    # Judge client management
    # ---------------------------------------------------------------------
    def _get_judge_client(self) -> Any:
        """Instantiate (once) and return the configured judge client."""

        if self._judge_client is not None:
            return self._judge_client

        factory = registry.factory(REGISTRY_GROUP_MODEL_CLIENTS, self.judge_client_name)
        client = factory(**self.judge_params)

        if getattr(client, "info", None) is None:
            client.info = ModelClientInfo(
                name=self.judge_client_name,
                provider="unknown",
                capabilities={"chat"},
            )

        if self._cache_config.enabled and is_chat_client(client):
            paths = self._paths or Paths.default()
            manager = get_cache_manager(paths, self._cache_config)
            if manager.enabled:
                client = CachingChatClient(
                    delegate=client,
                    manager=manager,
                    namespace=self._cache_namespace_prefix,
                    ttl_seconds=self._cache_config.ttl_seconds,
                )

        self._judge_client = client
        return client

    def _invoke_judge_with_telemetry(
        self, messages: Sequence[Message]
    ) -> tuple[Any, TurnTelemetry | None]:
        """Invoke judge client and return response with telemetry.

        Returns:
            Tuple of (response, telemetry) where telemetry is tagged with component="judge"
        """
        client = self._get_judge_client()
        response = client.invoke(messages=messages)

        # Extract and tag telemetry
        provider = getattr(client.info, "provider", None) if hasattr(client, "info") else None
        telemetry = usage_to_turn_telemetry(response.usage, provider=provider)

        if telemetry is not None:
            if telemetry.metadata is None:
                telemetry.metadata = {}
            telemetry.metadata["component"] = "judge"

            # Add judge configuration details for rich telemetry
            telemetry.metadata["judge_client_name"] = self.judge_client_name
            telemetry.metadata["metric_name"] = self.info.name

            # Add judge model parameters if available
            if self.judge_params:
                # Extract model-related parameters
                if "model" in self.judge_params:
                    telemetry.metadata["model"] = self.judge_params["model"]
                if "model_name" in self.judge_params:
                    telemetry.metadata["model_name"] = self.judge_params["model_name"]
                if "model_import" in self.judge_params:
                    telemetry.metadata["model_import"] = self.judge_params["model_import"]
                if "model_kwargs" in self.judge_params and isinstance(
                    self.judge_params["model_kwargs"], Mapping
                ):
                    model_kwargs = self.judge_params["model_kwargs"]
                    if "model" in model_kwargs:
                        telemetry.metadata["model"] = model_kwargs["model"]
                    if "azure_deployment" in model_kwargs:
                        telemetry.metadata["azure_deployment"] = model_kwargs["azure_deployment"]
                    if "temperature" in model_kwargs:
                        telemetry.metadata["temperature"] = model_kwargs["temperature"]

        return response, telemetry

    # ---------------------------------------------------------------------
    # Conversation utilities
    # ---------------------------------------------------------------------
    def _format_conversation(self, messages: Sequence[Message]) -> str:
        """Render a sequence of messages into a deterministic transcript."""

        lines: list[str] = []
        for message in messages:
            lines.append(f"{message.role.value.upper()}: {message.content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Randomness helpers
    # ------------------------------------------------------------------
    def _rng_seed(self, *, episode_id: str, base_seed: int, salt: str) -> int:
        """Derive a deterministic seed for the given episode and salt."""

        token = f"{base_seed}:{self.info.name}:{episode_id}:{salt}".encode()
        digest = hashlib.sha256(token).hexdigest()
        return int(digest[:16], 16)

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
    def _aggregate_scalar_metric(
        self,
        values: Sequence[MetricValue],
        *,
        extras: dict[str, Any] | None = None,
    ) -> MetricAggregate:
        """Aggregate scalar episode scores with optional bootstrap."""

        all_scores: list[float] = []
        for value in values:
            all_scores.extend(value.values)

        extras_payload = dict(extras or {})

        if not all_scores:
            return MetricAggregate(
                metric_name=self.info.name,
                mean=0.0,
                standard_deviation=0.0,
                confidence_interval=0.0,
                sample_size=0,
                extras=extras_payload,
            )

        use_bootstrap = (
            self._bootstrap_enabled and len(all_scores) > 1 and self.bootstrap_config.iterations > 0
        )

        if use_bootstrap:
            result = bootstrap_mean(all_scores, self.bootstrap_config)
            mean_score = result.mean
            stdev_score = result.standard_deviation
            ci_lower = result.ci_lower
            ci_upper = result.ci_upper
            extras_payload.setdefault("bootstrap", {})
            extras_payload["bootstrap"] = {
                "iterations": self.bootstrap_config.iterations,
                "confidence": self.bootstrap_config.confidence,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
            ci_half_width = (ci_upper - ci_lower) / 2.0
        else:
            mean_score, stdev_score, ci_half_width = mean_stdev_ci(
                all_scores,
                confidence=self.bootstrap_config.confidence,
            )
            extras_payload["bootstrap"] = None

        extras_payload.setdefault("count", len(all_scores))

        return MetricAggregate(
            metric_name=self.info.name,
            mean=mean_score,
            standard_deviation=stdev_score,
            confidence_interval=ci_half_width,
            sample_size=len(all_scores),
            extras=extras_payload,
        )


__all__ = [
    "JudgeMetricMixin",
]
