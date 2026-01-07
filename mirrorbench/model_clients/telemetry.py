"""Telemetry helpers shared by model clients."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from time import perf_counter
from typing import Any, Protocol

from mirrorbench.core.models.messages import TurnTelemetry
from mirrorbench.core.models.registry import ModelClientInfo
from mirrorbench.model_clients.utils import coerce_float, coerce_int


class SupportsUsage(Protocol):
    usage: Mapping[str, Any] | None


def normalize_usage(info: ModelClientInfo, usage: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a normalized copy of usage statistics emitted by a client."""

    normalized: dict[str, Any] = dict(usage or {})
    normalized.setdefault("provider", info.provider)
    if info.capabilities:
        normalized.setdefault("capabilities", sorted(info.capabilities))
    return normalized


def usage_to_turn_telemetry(
    usage: Mapping[str, Any] | None,
    *,
    provider: str | None = None,
) -> TurnTelemetry | None:
    """Convert a usage payload into :class:`TurnTelemetry` when possible."""

    if usage is None:
        return None

    telemetry = TurnTelemetry(
        time_to_first_token=coerce_float(usage.get("time_to_first_token")),
        time_per_output_token=coerce_float(usage.get("time_per_output_token")),
        total_response_time=coerce_float(usage.get("total_response_time")),
        tokens_input=coerce_int(usage.get("tokens_input")),
        tokens_output=coerce_int(usage.get("tokens_output")),
        cost_usd=coerce_float(usage.get("cost_usd")),
        provider=provider,
    )

    known_keys = {
        "time_to_first_token",
        "time_per_output_token",
        "total_response_time",
        "tokens_input",
        "tokens_output",
        "cost_usd",
    }
    metadata = {str(key): value for key, value in usage.items() if key not in known_keys}
    telemetry.metadata = metadata
    return telemetry


def invoke_with_telemetry(
    info: ModelClientInfo, call: Callable[[], SupportsUsage]
) -> SupportsUsage:
    """Execute ``call`` and annotate its response usage with timing metrics."""

    start = perf_counter()
    response = call()
    elapsed = perf_counter() - start
    usage = normalize_usage(info, response.usage)
    usage.setdefault("time_to_first_token", None)
    usage.setdefault("time_per_output_token", None)
    usage["total_response_time"] = elapsed
    response.usage = usage
    return response


def stream_with_telemetry(iterator: Iterator[Any], *, info: ModelClientInfo) -> Iterator[Any]:
    """Yield streaming chunks while attaching telemetry measurements."""

    start = perf_counter()
    first_token_time: float | None = None
    total_tokens: int | None = None
    telemetry: dict[str, Any] = {
        "provider": info.provider,
        "tokens_input": None,
        "tokens_output": None,
        "time_to_first_token": None,
        "time_per_output_token": None,
        "total_response_time": None,
    }

    for chunk in iterator:
        now = perf_counter()
        if first_token_time is None:
            first_token_time = now
            telemetry["time_to_first_token"] = now - start
        usage = _usage_from_chunk(chunk)
        if usage is not None:
            tokens_input = coerce_int(usage.get("tokens_input"))
            if tokens_input is not None:
                telemetry["tokens_input"] = tokens_input
            tokens_output = coerce_int(usage.get("tokens_output"))
            if tokens_output is not None:
                telemetry["tokens_output"] = tokens_output
                total_tokens = tokens_output
        if hasattr(chunk, "telemetry"):
            chunk.telemetry = telemetry
        yield chunk

    end = perf_counter()
    telemetry["total_response_time"] = end - start
    if first_token_time is not None and total_tokens:
        telemetry["time_per_output_token"] = (end - first_token_time) / total_tokens


def _usage_from_chunk(chunk: Any) -> Mapping[str, Any] | None:
    raw = getattr(chunk, "raw", None)
    if isinstance(raw, Mapping):
        usage = raw.get("usage")
        if isinstance(usage, Mapping):
            return usage
    return None


__all__ = ["SupportsUsage", "invoke_with_telemetry", "normalize_usage", "stream_with_telemetry"]
