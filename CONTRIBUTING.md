# Contributing to MirrorBench

Welcome! This guide explains how to set up a local development environment for MirrorBench and run the core quality checks. The project targets Python 3.12+.

## 1. Prerequisites

- **Python 3.12** available on your PATH (`python3 --version` should report 3.12.x)
- **Git** and **VS Code** (or your preferred editor)
- Optional tooling: [`uv`](https://docs.astral.sh/uv/), [`pre-commit`](https://pre-commit.com/), [`nox`](https://nox.thea.codes/) for automation helpers

## 2. Clone the repository

```bash
git clone https://github.com/SAP/mirrorbench.git
cd mirrorbench
```

## 3. Create and activate a virtual environment

### Preferred: uv-managed environment

```bash
uv venv
source .venv/bin/activate
```

`uv` provisions `.venv/` and speeds up future installs. VS Code usually detects the interpreter automatically; if not, open the Command Palette → “Python: Select Interpreter” and pick `.venv`.

### Alternative: built-in venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

Both approaches integrate cleanly with the remaining steps—choose whichever suits your workflow.

## 4. Install dependencies

Install the package in editable mode (only core runtime dependencies are required today):

```bash
uv pip install -e .
```

> Note: `uv pip install -e .` performs an **editable install**. Any code changes you make in this repository take effect immediately when you import `mirrorbench` or run the `mirrorbench` CLI—no reinstall required.

Install development tooling via the `dev` optional dependency group:

```bash
uv pip install -e ".[dev]"
```

If you prefer standard tooling, swap `uv pip` for `python -m pip` (e.g., `python -m pip install -e ".[dev]"`).

## 5. Run the quality checks

```bash
# type checking
mypy mirrorbench

# linting + formatting checks
ruff check mirrorbench tests
ruff format --check mirrorbench tests

# unit tests
pytest -m "not integration" tests
```

To keep the project clean automatically, enable pre-commit hooks:

```bash
pre-commit install
pre-commit run --all-files
```

## 6. Development workflow tips

- Use `make install`, `make lint`, `make typecheck`, and `make test` for shorthand commands.
- `nox -s tests` runs the automated session matrix if you installed `nox`.
- When editing Pydantic models or registries, keep typings strict; the project enforces `mypy --strict`.

## 7. Code style & docstrings

Follow these conventions so documentation and tooling remain consistent:

- **Classes**: include a class-level docstring summarizing responsibility. Document noteworthy attributes using either inline docstrings or concise comments.
- **Methods**: public methods require docstrings explaining parameters, behavior, and return values. Protected/private helpers may omit docstrings unless behavior is non-obvious.
- **Type hints**: annotate all function arguments and return types (`-> None` when applicable). For optional values, prefer `typing.Optional[T]` (e.g., `Optional[str]`) for readability.
- **Formatting example**:

  ```python
  class ExampleResource:
      """Manage Example resources and expose CRUD operations."""

      resource_id: str
      """Server-generated identifier."""

      def __init__(self, resource_id: str) -> None:
          self.resource_id = resource_id

      def _validate(self) -> None:
          """Internal helper; raises ValueError if state is inconsistent."""

      def refresh(self) -> None:
          """Reload the resource from the backend."""
  ```

These guidelines keep the codebase clean and enable easy navigation for contributors.

## 7. Submitting changes

1. Ensure all checks pass (`mypy`, `ruff`, `pytest`).
2. Update or add tests and documentation as needed.
3. Create a pull request with a clear description of your changes.

## Code of Conduct

All members of the project community must abide by the [SAP Open Source Code of Conduct](https://github.com/SAP/.github/blob/main/CODE_OF_CONDUCT.md).
Only by respecting each other we can develop a productive, collaborative community.
Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting [a project maintainer](REUSE.toml).

## Engaging in Our Project

We use GitHub to manage reviews of pull requests.

* If you are a new contributor, see: [Steps to Contribute](#steps-to-contribute)

* Before implementing your change, create an issue that describes the problem you would like to solve or the code that should be enhanced. Please note that you are willing to work on that issue.

* The team will review the issue and decide whether it should be implemented as a pull request. In that case, they will assign the issue to you. If the team decides against picking up the issue, the team will post a comment with an explanation.

## Steps to Contribute

Should you wish to work on an issue, please claim it first by commenting on the GitHub issue that you want to work on. This is to prevent duplicated efforts from other contributors on the same issue.

If you have questions about one of the issues, please comment on them, and one of the maintainers will clarify.

## Contributing Code or Documentation

You are welcome to contribute code in order to fix a bug or to implement a new feature that is logged as an issue.

The following rule governs code contributions:

* Contributions must be licensed under the [Apache 2.0 License](./LICENSE).
* Due to legal reasons, contributors will be asked to accept a Developer Certificate of Origin (DCO) when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).
* Contributions must follow our [guidelines on AI-generated code](https://github.com/SAP/.github/blob/main/CONTRIBUTING_USING_GENAI.md) in case you are using such tools.

## Issues and Planning

* We use GitHub issues to track bugs and enhancement requests.

* Please provide as much context as possible when you open an issue. The information you provide must be comprehensive enough to reproduce that issue for the assignee.
