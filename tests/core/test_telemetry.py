from __future__ import annotations

import importlib

import pytest

from mirrorbench.core.config import ObservabilityConfig


@pytest.fixture()
def telemetry_module():
    from mirrorbench.core import telemetry

    module = importlib.reload(telemetry)
    yield module
    importlib.reload(telemetry)


def test_telemetry_returns_noop_when_disabled(telemetry_module) -> None:
    config = ObservabilityConfig(tracing_enabled=False, metrics_enabled=False)
    telemetry_module.configure_telemetry_from_config(config)

    tracer = telemetry_module.get_tracer("mirrorbench.test")
    meter = telemetry_module.get_meter("mirrorbench.test")

    assert isinstance(tracer, telemetry_module._NullTracer)
    assert isinstance(meter, telemetry_module._NullMeter)


@pytest.mark.skipif(
    not importlib.import_module("mirrorbench.core.telemetry").OTEL_AVAILABLE,
    reason="OpenTelemetry not installed",
)
def test_telemetry_enables_opentelemetry_when_available(telemetry_module) -> None:
    config = ObservabilityConfig(tracing_enabled=True, metrics_enabled=False, otel_exporter="noop")
    telemetry_module.configure_telemetry_from_config(config)

    tracer = telemetry_module.get_tracer("mirrorbench.test")
    meter = telemetry_module.get_meter("mirrorbench.test")

    assert not isinstance(tracer, telemetry_module._NullTracer)
    assert isinstance(meter, telemetry_module._NullMeter)

    with tracer.start_as_current_span("span") as span:
        span.set_attribute("mirrorbench.test", True)

    counter = meter.create_counter("mirrorbench.test.counter")
    counter.add(1)
