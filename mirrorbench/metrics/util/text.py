"""Shared tokenization utilities for metrics using tiktoken."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import cast


def tokenize(text: str, *, model: str = "gpt-4o") -> list[int]:
    """Tokenize text using tiktoken for the specified model.

    This provides accurate tokenization that reflects how modern LLMs
    process text. The default model is gpt-4o, but users can override this
    to match their specific use case.

    Args:
        text: The input text to tokenize.
        model: The model name to get the encoding for (default: "gpt-4o").
            Common options: "gpt-4o", "gpt-4", "gpt-3.5-turbo", "text-embedding-ada-002".

    Returns:
        List of token IDs from the tiktoken encoding.

    Examples:
        >>> tokens = tokenize("Hello world!")
        >>> len(tokens)
        3
        >>> tokens_gpt4 = tokenize("Hello world!", model="gpt-4")
        >>> len(tokens_gpt4)
        3
    """
    tiktoken_module = importlib.import_module("tiktoken")
    encoding = tiktoken_module.encoding_for_model(model)
    encoded = encoding.encode(text)
    return cast(list[int], encoded)


def compute_token_ngrams(tokens: Sequence[int], n: int) -> list[tuple[int, ...]]:
    """Generate n-grams from a sequence of token IDs using NLTK.

    Uses NLTK's battle-tested ngrams implementation for precision and reliability.
    For a sequence [a, b, c, d] with n=2, produces [(a,b), (b,c), (c,d)].

    Args:
        tokens: Sequence of token IDs from tiktoken.
        n: N-gram size (e.g., 1 for unigrams, 2 for bigrams).

    Returns:
        List of n-gram tuples of token IDs. Returns empty list if n < 1 or n > len(tokens).

    Examples:
        >>> tokens = [100, 200, 300]
        >>> compute_token_ngrams(tokens, 1)
        [(100,), (200,), (300,)]
        >>> compute_token_ngrams(tokens, 2)
        [(100, 200), (200, 300)]
    """
    if n < 1 or n > len(tokens):
        return []
    nltk_module = importlib.import_module("nltk.util")
    ngrams_fn = nltk_module.ngrams
    return list(ngrams_fn(tokens, n))


def distinct_n(tokens: Sequence[int], n: int) -> float:
    """Compute distinct-n metric (ratio of unique n-grams to total n-grams).

    This is a common diversity metric in NLP that measures lexical variety
    based on token IDs from tiktoken.

    Args:
        tokens: Sequence of token IDs from tiktoken.
        n: N-gram size.

    Returns:
        Ratio of unique n-grams to total n-grams, or 0.0 if no n-grams exist.

    Examples:
        >>> tokens = [100, 200, 100, 200]  # Repeated pattern
        >>> distinct_n(tokens, 1)
        0.5
        >>> distinct_n(tokens, 2)
        0.667
    """
    ngrams = compute_token_ngrams(tokens, n)
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


__all__ = [
    "compute_token_ngrams",
    "distinct_n",
    "tokenize",
]
