"""Optional OpenTelemetry integration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from types import TracebackType
from typing import Any, Literal, cast

from mirrorbench.core.config import ObservabilityConfig


def _optional_import(module: str) -> Any:
    try:
        return import_module(module)
    except ImportError:  # pragma: no cover - optional dependency missing
        return None


_metrics_mod = _optional_import("opentelemetry.metrics")
_trace_mod = _optional_import("opentelemetry.trace")
_sdk_resources = _optional_import("opentelemetry.sdk.resources")
_sdk_trace = _optional_import("opentelemetry.sdk.trace")
_sdk_trace_export = _optional_import("opentelemetry.sdk.trace.export")
_sdk_metrics = _optional_import("opentelemetry.sdk.metrics")
_sdk_metrics_export = _optional_import("opentelemetry.sdk.metrics.export")
_otlp_trace_export = _optional_import("opentelemetry.exporter.otlp.proto.http.trace_exporter")
_otlp_metric_export = _optional_import("opentelemetry.exporter.otlp.proto.http.metric_exporter")

OTEL_AVAILABLE = all(
    module is not None
    for module in (
        _metrics_mod,
        _trace_mod,
        _sdk_resources,
        _sdk_trace,
        _sdk_trace_export,
        _sdk_metrics,
        _sdk_metrics_export,
    )
)

metrics = cast(Any, _metrics_mod)
trace = cast(Any, _trace_mod)
Resource = None if _sdk_resources is None else getattr(_sdk_resources, "Resource", None)
TracerProvider = None if _sdk_trace is None else getattr(_sdk_trace, "TracerProvider", None)
MeterProvider = None if _sdk_metrics is None else getattr(_sdk_metrics, "MeterProvider", None)
BatchSpanProcessor = (
    None if _sdk_trace_export is None else getattr(_sdk_trace_export, "BatchSpanProcessor", None)
)
ConsoleSpanExporter = (
    None if _sdk_trace_export is None else getattr(_sdk_trace_export, "ConsoleSpanExporter", None)
)
ConsoleMetricExporter = (
    None
    if _sdk_metrics_export is None
    else getattr(_sdk_metrics_export, "ConsoleMetricExporter", None)
)
PeriodicExportingMetricReader = (
    None
    if _sdk_metrics_export is None
    else getattr(_sdk_metrics_export, "PeriodicExportingMetricReader", None)
)
OTLPSpanExporter = (
    None if _otlp_trace_export is None else getattr(_otlp_trace_export, "OTLPSpanExporter", None)
)
OTLPMetricExporter = (
    None
    if _otlp_metric_export is None
    else getattr(_otlp_metric_export, "OTLPMetricExporter", None)
)


@dataclass(slots=True)
class _NullSpan:
    def __enter__(self) -> _NullSpan:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        return False

    def set_attribute(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - noop
        return None


class _NullTracer:
    def start_as_current_span(self, name: str, **kwargs: Any) -> _NullSpan:
        return _NullSpan()


class _NullCounter:
    def add(self, amount: int | float, **kwargs: Any) -> None:  # pragma: no cover - noop
        return None


class _NullMeter:
    def create_counter(
        self, name: str, *, unit: str = "1", description: str | None = None
    ) -> _NullCounter:
        return _NullCounter()


_TRACING_ENABLED = False
_METRICS_ENABLED = False
_TELEMETRY_CONFIGURED = False
_RESOURCE: Any = None


def _resource() -> Any:
    global _RESOURCE  # noqa: PLW0603
    if _RESOURCE is None and OTEL_AVAILABLE and Resource is not None:
        _RESOURCE = Resource.create({"service.name": "mirrorbench"})
    return _RESOURCE


def configure_telemetry_from_config(config: ObservabilityConfig) -> None:
    global _TRACING_ENABLED, _METRICS_ENABLED, _TELEMETRY_CONFIGURED  # noqa: PLW0603

    if _TELEMETRY_CONFIGURED:
        return

    if not OTEL_AVAILABLE or (not config.tracing_enabled and not config.metrics_enabled):
        _TELEMETRY_CONFIGURED = True
        return

    exporter_pref = os.getenv("MIRRORBENCH_OTEL_EXPORTER", config.otel_exporter).lower()
    endpoint = os.getenv("MIRRORBENCH_OTEL_ENDPOINT", config.otel_endpoint or "").strip()

    if (
        config.tracing_enabled
        and trace is not None
        and TracerProvider is not None
        and BatchSpanProcessor is not None
    ):
        resource = _resource()
        tracer_provider = (
            TracerProvider(resource=resource) if resource is not None else TracerProvider()
        )
        span_exporter = _resolve_span_exporter(exporter_pref, endpoint)
        if span_exporter is not None:
            tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        if hasattr(trace, "set_tracer_provider"):
            trace.set_tracer_provider(tracer_provider)
        _TRACING_ENABLED = True

    if (
        config.metrics_enabled
        and metrics is not None
        and MeterProvider is not None
        and PeriodicExportingMetricReader is not None
    ):
        metric_exporter = _resolve_metric_exporter(exporter_pref, endpoint)
        if metric_exporter is not None:
            resource = _resource()
            reader = PeriodicExportingMetricReader(metric_exporter)
            if resource is not None:
                meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
            else:
                meter_provider = MeterProvider(metric_readers=[reader])
            if hasattr(metrics, "set_meter_provider"):
                metrics.set_meter_provider(meter_provider)
            _METRICS_ENABLED = True

    _TELEMETRY_CONFIGURED = True


def _resolve_span_exporter(exporter: str, endpoint: str | None) -> Any:
    if exporter == "noop":
        return None
    if exporter == "stdout" or not exporter:
        return ConsoleSpanExporter() if ConsoleSpanExporter is not None else None
    if exporter == "otlp" and OTLPSpanExporter is not None:
        kwargs: dict[str, Any] = {}
        if endpoint:
            kwargs["endpoint"] = endpoint
        return OTLPSpanExporter(**kwargs)
    return ConsoleSpanExporter() if ConsoleSpanExporter is not None else None


def _resolve_metric_exporter(exporter: str, endpoint: str | None) -> Any:
    if exporter == "noop":
        return None
    if exporter == "stdout" or not exporter:
        return ConsoleMetricExporter() if ConsoleMetricExporter is not None else None
    if exporter == "otlp" and OTLPMetricExporter is not None:
        kwargs: dict[str, Any] = {}
        if endpoint:
            kwargs["endpoint"] = endpoint
        return OTLPMetricExporter(**kwargs)
    return ConsoleMetricExporter() if ConsoleMetricExporter is not None else None


def get_tracer(name: str) -> Any:
    if OTEL_AVAILABLE and _TRACING_ENABLED and trace is not None and hasattr(trace, "get_tracer"):
        return trace.get_tracer(name)
    return _NullTracer()


def get_meter(name: str) -> Any:
    if (
        OTEL_AVAILABLE
        and _METRICS_ENABLED
        and metrics is not None
        and hasattr(metrics, "get_meter_provider")
    ):
        return metrics.get_meter_provider().get_meter(name)
    return _NullMeter()


__all__ = ["configure_telemetry_from_config", "get_meter", "get_tracer"]
