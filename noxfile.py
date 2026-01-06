"""Automation sessions for linting, testing, and type checking."""

from __future__ import annotations

import pathlib

import nox

PACKAGE = "mirrorbench"
PYTHON_VERSIONS = ["3.12"]
ROOT = pathlib.Path(__file__).parent


@nox.session
def lint(session: nox.Session) -> None:
    session.install("ruff>=0.1.9")
    session.run("ruff", "check", "mirrorbench", "tests")


@nox.session
def format(session: nox.Session) -> None:
    session.install("ruff>=0.1.9")
    session.run("ruff", "format", "mirrorbench", "tests")


@nox.session
def typecheck(session: nox.Session) -> None:
    session.install("mypy>=1.7", "types-PyYAML>=6.0.12", "types-requests>=2.31")
    session.install("-e", ".")
    session.run("mypy", "mirrorbench")


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("pytest", "-m", "not integration", "tests")
