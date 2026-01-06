"""Helpers for parsing JSON responses."""

from __future__ import annotations

import json
from typing import Any, cast

from pydantic import BaseModel


def parse_json(
    response_text: str,
    *,
    validate_pydantic_object: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """Parse JSON from a response string and optionally validate with Pydantic."""

    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    cleaned_text = cleaned_text.strip()

    try:
        payload = json.loads(cleaned_text)
    except json.JSONDecodeError as exc:  # pragma: no cover - error path
        msg = f"Response is not valid JSON: {response_text}"
        raise ValueError(msg) from exc

    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")

    if validate_pydantic_object is not None:
        try:
            validate_pydantic_object.model_validate(payload)
        except Exception as exc:  # pragma: no cover - validation error path
            msg = (
                "JSON payload does not match expected pydantic schema.\n"
                f"Response text: {response_text}\n"
                f"Error: {exc}"
            )
            raise ValueError(msg) from exc

    return cast(dict[str, Any], payload)


__all__ = [
    "parse_json",
]
