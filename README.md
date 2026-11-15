# ReviewLens

Sentiment classification of Persian customer reviews (positive / negative), a study of
**semi-supervised learning with few labels**, and a **tool-calling AI agent** that explains
the results.

> Research question: when only a small fraction of the data is labeled, do semi-supervised
> methods outperform purely supervised models trained on the labeled fraction alone?

**Status:** phase 1 (scaffold and design). The full README (results, run modes, findings)
is written in phase 8.

## Setup

Requirements: Python 3.12 (3.11+ is supported) and GNU make
(on Windows: `winget install ezwinports.make`).

```bash
make install      # creates .venv and installs the package with dev tools
make test         # pytest
make lint         # ruff + mypy
```

Without make, run the same steps directly:

```bash
py -3.12 -m venv .venv                      # Linux/macOS: python3 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"   # Linux/macOS: .venv/bin/python
.venv\Scripts\python -m pytest
```

## CLI

```bash
reviewlens --help
reviewlens show-config
reviewlens --experiment configs/experiment/quick.yaml show-config
```

## Configuration

All settings live in `configs/default.yaml`. Experiment files in `configs/experiment/`
override only the keys they mention. API keys are never stored in config:
see `.env.example` for the environment variable names.

## Project layout

See `CLAUDE.md` for the full specification and `docs/` for the design sheet,
the prompt notebook and the Persian phase explanations.
