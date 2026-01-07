"""Lightweight logging helpers for preprocessing scripts."""

from __future__ import annotations

import logging
from typing import Dict

_LOGGERS: Dict[str, logging.Logger] = {}
_CONFIGURED = False


def get_logger(name: str = "mirrorbench.preprocess") -> logging.Logger:
    """Return a module-level logger configured with a reasonable formatter."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        )
        _CONFIGURED = True
    if name not in _LOGGERS:
        _LOGGERS[name] = logging.getLogger(name)
    return _LOGGERS[name]
