# ReviewLens

Sentiment classification of Persian customer reviews (positive / negative), a study of
**semi-supervised learning with few labels**, and a **tool-calling AI agent** that explains
the results.

> Research question: when only a small fraction of the data is labeled, do semi-supervised
> methods outperform purely supervised models trained on the labeled fraction alone?

**Status:** phase 2 (data). The full README (results, run modes, findings)
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

## Data

Snappfood food-delivery comments (binary HAPPY / SAD, ~69.5k reviews) from
[ParsiAI/snappfood-sentiment-analysis](https://huggingface.co/datasets/ParsiAI/snappfood-sentiment-analysis),
pinned to one revision in `configs/default.yaml`. The official `test.csv` is the fixed test
set; `train.csv` + `validation.csv` form the training pool.

```bash
reviewlens data download   # raw CSVs -> data/raw/ (never committed)
reviewlens data prepare    # normalize, validate, deduplicate -> data/processed/
```

Exploration: `notebooks/01_eda.ipynb` (install the extra with `pip install -e ".[notebook]"`).

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
