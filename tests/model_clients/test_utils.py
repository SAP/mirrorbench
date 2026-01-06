from __future__ import annotations

import pytest

from mirrorbench.model_clients.utils import coerce_float, coerce_int


def test_coerce_int_handles_valid_inputs() -> None:
    assert coerce_int(5) == 5  # noqa: PLR2004
    assert coerce_int("12") == 12  # noqa: PLR2004
    assert coerce_int(3.7) == 3  # noqa: PLR2004


def test_coerce_int_returns_none_for_invalid_values() -> None:
    assert coerce_int("not-an-int") is None
    assert coerce_int(None) is None


def test_coerce_float_handles_valid_inputs() -> None:
    assert coerce_float(5.5) == 5.5  # noqa: PLR2004
    assert coerce_float("3.14") == pytest.approx(3.14)
    assert coerce_float(10) == 10.0  # noqa: PLR2004


def test_coerce_float_returns_none_for_invalid_values() -> None:
    assert coerce_float("abc") is None
    assert coerce_float(None) is None
