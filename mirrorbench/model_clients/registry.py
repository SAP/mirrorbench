"""Register built-in model clients."""

from __future__ import annotations

# Import subpackages to trigger registration side effects
from mirrorbench.model_clients import (
    langchain,  # noqa: F401
    openai,  # noqa: F401
)

__all__: list[str] = []
