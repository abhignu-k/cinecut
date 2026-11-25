# Contributing to cinecut

Thanks for contributing to `cinecut`!

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Ensure `ffmpeg` and `ffprobe` are installed and available on your `PATH`.

## Quality Gates

Before committing or opening a pull request, run:

```bash
ruff format --check src tests
ruff check src tests
mypy src tests
CI=true pytest --cov=cinecut --cov-fail-under=80
```

## Code Style & Architecture
- Follow `conductor/code_styleguides/python.md` (80-char line length, Google docstrings, explicit type annotations).
- Keep all external AI/FFmpeg tests offline-safe using `pytest` mocks or FFmpeg `lavfi` synthetic media.
