.PHONY: install lint format typecheck test test-coverage precommit

VENV ?= .venv
PYTHON := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python)
PYTEST := $(PYTHON) -m pytest

install:
	uv pip install -e .[dev]

lint:
	ruff check mirrorbench tests

format:
	ruff format mirrorbench tests

typecheck:
	mypy mirrorbench

test:
	$(PYTEST) -m "not integration" tests

test-integration:
	$(PYTEST) -m integration tests

test-coverage:
	$(PYTEST) --cov=mirrorbench --cov-report=term-missing tests

precommit:
	uv run pre-commit run --all-files
