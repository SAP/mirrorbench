"""Embedding client plumbing for few-shot retrieval.

Fill in `get_embedding_client` with a real embedding model. By default the
function returns ``None`` so the preprocessing pipeline will skip few-shot
example generation until an embedding backend is provided.
"""

from __future__ import annotations

from typing import Protocol, Sequence


class EmbeddingClient(Protocol):
    """Protocol describing the minimal embedding client interface."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return vector embeddings for the provided texts."""


def get_embedding_client() -> EmbeddingClient | None:
    """Return an embedding client implementation or ``None`` when unavailable."""

    # Replace below code with your own embedding client initialization.
    from cached_scripts import get_embedding_model
    return get_embedding_model(model_name="text-embedding-3-large")
