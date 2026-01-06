"""Exceptions raised by model clients."""

from __future__ import annotations

from mirrorbench.core.models.errors import RunnerError


class ModelClientError(RunnerError):
    """Base error for model client failures."""


class AuthenticationError(ModelClientError):
    """Raised when credentials are missing or invalid."""


class TransportError(ModelClientError):
    """Raised for network/transport level failures."""


__all__ = [
    "AuthenticationError",
    "ModelClientError",
    "TransportError",
]
