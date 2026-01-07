"""Structlog configuration helpers for MirrorBench."""

from __future__ import annotations

import logging
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import structlog
from structlog import contextvars as structlog_contextvars

from mirrorbench.core.config import ObservabilityConfig

_LOGGING_CONFIGURED = False


def configure_logging_from_config(config: ObservabilityConfig, *, force: bool = False) -> None:
    """Configure structlog/std logging according to the provided observability config."""

    global _LOGGING_CONFIGURED  # noqa: PLW0603
    if _LOGGING_CONFIGURED and not force:
        return

    log_format = os.getenv("MIRRORBENCH_LOG_FORMAT")
    log_json = config.log_json if log_format is None else log_format.lower() != "text"

    log_level = os.getenv("MIRRORBENCH_LOG_LEVEL", config.log_level).upper()
    try:
        numeric_level = getattr(logging, log_level)
    except AttributeError:
        numeric_level = logging.INFO

    destination = os.getenv("MIRRORBENCH_LOG_DESTINATION", config.log_destination or "").strip()
    handler: logging.Handler
    if destination:
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(dest_path)
    else:
        handler = logging.StreamHandler(sys.stderr)

    if log_json:
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    logging.basicConfig(level=numeric_level, handlers=[handler], force=True)

    processors: list[Any] = [
        structlog_contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        try:
            console_module = import_module("structlog.dev")
            console_renderer = getattr(console_module, "ConsoleRenderer", None)
        except ImportError:  # pragma: no cover - fallback to JSON when renderer missing
            console_renderer = None
        if console_renderer is not None:
            processors.append(console_renderer())
        else:
            processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    structlog_contextvars.clear_contextvars()
    _LOGGING_CONFIGURED = True


__all__ = ["configure_logging_from_config"]
