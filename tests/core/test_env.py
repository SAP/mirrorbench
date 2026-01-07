from __future__ import annotations

import pytest

from mirrorbench.core import env as env_module
from mirrorbench.core.env import get_env, load_env_map, require_env, resolve_any


def test_get_env_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MB_TEST_KEY", "value")
    assert get_env("MB_TEST_KEY") == "value"


def test_get_env_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MB_TEST_MISSING", raising=False)
    assert get_env("MB_TEST_MISSING", default="fallback") == "fallback"


def test_require_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MB_TEST_REQUIRED", raising=False)
    with pytest.raises(RuntimeError):
        require_env("MB_TEST_REQUIRED")


def test_resolve_any_prefers_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MB_ENV_SECOND", "second")
    monkeypatch.setenv("MB_ENV_FIRST", "first")
    assert resolve_any(["MB_ENV_FIRST", "MB_ENV_SECOND"]) == "first"


def test_resolve_any_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MB_ENV_NONE", raising=False)
    with pytest.raises(RuntimeError):
        resolve_any(["MB_ENV_NONE"], required=True)


def test_load_env_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MB_ENV_A", "a")
    monkeypatch.delenv("MB_ENV_B", raising=False)
    mapping = load_env_map([("MB_ENV_A", None), ("MB_ENV_B", "b")])
    assert mapping == {"MB_ENV_A": "a", "MB_ENV_B": "b"}


def test_get_env_loads_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path):
    env_module._DOTENV_LOADED = False  # type: ignore[attr-defined]
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("MB_ENV_FILE=from_dotenv", encoding="utf-8")
    monkeypatch.delenv("MB_ENV_FILE", raising=False)
    value = env_module.get_env("MB_ENV_FILE")
    assert value == "from_dotenv"
    env_module._DOTENV_LOADED = False  # reset for other tests
