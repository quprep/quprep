# Contributing to QuPrep

Thank you for your interest in contributing. QuPrep is a focused tool — contributions that keep it simple, correct, and composable are most welcome.

## Before you start

- Check [open issues](https://github.com/quprep/quprep/issues) to avoid duplicate work.
- For large changes, open an issue first to discuss the approach.
- QuPrep's scope is intentionally narrow. Features outside the ingest → clean → reduce → normalize → encode → export pipeline are unlikely to be accepted.

## Development setup

Install [uv](https://docs.astral.sh/uv/) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and set up the project:

```bash
git clone https://github.com/quprep/quprep.git
cd quprep
uv sync --extra dev
```

Run tests:

```bash
uv run pytest
```

Lint and format:

```bash
uv run ruff check .
uv run ruff format .
```

## Pull request guidelines

- Keep PRs focused. One feature or fix per PR.
- Add or update tests for any changed behaviour.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Docstrings should include the mathematical formulation where relevant (e.g., for encoders and normalizers).
- Target the `main` branch.

## Test requirements

- All tests must pass.
- New encoders require property-based tests (e.g., amplitude encoder must produce unit-norm states).
- Target ≥ 90% coverage for new modules.

## Code style

- Python ≥ 3.9, type hints encouraged.
- `ruff` for linting and formatting (configured in `pyproject.toml`).
- No external dependencies beyond core (`numpy`, `scipy`, `pandas`, `scikit-learn`). Framework packages go in optional extras only.

## Reporting bugs

Use the [bug report template](https://github.com/quprep/quprep/issues/new?template=bug_report.md). Include a minimal reproducible example.

## Questions

Use [GitHub Discussions](https://github.com/quprep/quprep/discussions) for questions. Issues are for bugs and feature requests only.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be respectful.
