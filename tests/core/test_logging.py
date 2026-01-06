from __future__ import annotations

import json

import structlog

from mirrorbench.core.config import ObservabilityConfig
from mirrorbench.core.logging import configure_logging_from_config


def test_configure_logging_json(capsys) -> None:
    config = ObservabilityConfig(log_json=True, log_level="INFO")
    configure_logging_from_config(config, force=True)
    logger = structlog.get_logger("test").bind(test_id="123")
    logger.info("hello")
    out = capsys.readouterr().err.strip()
    payload = json.loads(out)
    assert payload["event"] == "hello"
    assert payload["test_id"] == "123"
    assert payload["level"] == "info"
