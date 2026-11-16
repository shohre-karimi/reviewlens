# ReviewLens

Sentiment classification of Persian customer reviews (negative / positive), a study of
**semi-supervised learning with few labels**, and a **tool-calling AI agent** that answers
questions about the model, the data and the results.

> **Research question:** when only a small fraction of the data is labeled, do
> semi-supervised methods outperform purely supervised models trained on the labeled
> fraction alone?

**Short answer on this dataset: no.** With 1 % to 20 % of the labels, no semi-supervised
method beat the best supervised model consistently; see [Findings](#findings).

University project (Python course + Machine Learning course). Persian explanations of every
phase are in [`docs/explanations/`](docs/explanations/).

---

## Contents

- [Quick start](#quick-start)
- [Findings](#findings)
- [Reproducing the results](#reproducing-the-results)
- [The app](#the-app)
- [The agent and its three run modes](#the-agent-and-its-three-run-modes)
- [Tests and code quality](#tests-and-code-quality)
- [Project layout](#project-layout)
- [Limitations](#limitations)

---

## Quick start

Requirements: Python 3.11+ (developed on 3.12) and GNU make
(Windows: `winget install ezwinports.make`).

```bash
make install        # .venv + package + dev tools + openai extra + pre-commit hooks
make test           # 255 tests
make lint           # ruff (lint + format check) + mypy --strict
make app            # Streamlit app on http://localhost:8501
```

Without make (Windows shown; on Linux/macOS use `python3` and `.venv/bin/`):

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,openai]"
.venv\Scripts\python -m pytest
.venv\Scripts\python -m streamlit run src/reviewlens/app/streamlit_app.py
```

Before running anything else, get the data and build the artifacts once
(see [Reproducing the results](#reproducing-the-results)).

---

## Findings

Data: Snappfood food-delivery reviews, 60,382 training and 9,032 test reviews, balanced
classes. The test set is fixed and never used for training or fitting. Labels in the
training set are hidden ("masked") so that only 1 %, 5 %, 10 %, 20 % or 100 % remain;
every method × fraction runs with 5 seeds (175 runs).

**Macro-F1 on the test set** (mean ± std over 5 seeds; best per column in bold):

| Method | 1 % | 5 % | 10 % | 20 % | 100 % |
|---|---|---|---|---|---|
| Linear SVM (calibrated) | **0.819 ± 0.003** | **0.841 ± 0.003** | 0.848 ± 0.003 | 0.851 ± 0.003 | **0.862** |
| Logistic Regression | 0.817 ± 0.004 | **0.841 ± 0.003** | **0.849 ± 0.002** | **0.853 ± 0.002** | **0.862** |
| Multinomial Naive Bayes | 0.815 ± 0.005 | 0.839 ± 0.002 | 0.843 ± 0.003 | 0.847 ± 0.003 | 0.852 |
| Random Forest | 0.794 ± 0.011 | 0.831 ± 0.003 | 0.838 ± 0.003 | 0.846 ± 0.003 | 0.856 |
| *Pseudo-Labeling (from scratch)* | 0.773 ± 0.009 | 0.833 ± 0.001 | 0.848 ± 0.001 | **0.853 ± 0.001** | **0.862** |
| *Self-Training (scikit-learn)* | 0.340 ± 0.013 | 0.810 ± 0.003 | 0.826 ± 0.002 | 0.844 ± 0.001 | **0.862** |
| *Label Spreading (kNN on LSA)* | 0.742 ± 0.008 | 0.774 ± 0.001 | 0.787 ± 0.003 | 0.799 ± 0.003 | 0.812 |

*Italic = semi-supervised.* Main figure: `artifacts/figures/default/macro_f1_vs_fraction.png`
(created by `reviewlens evaluate`, also shown in the app).

**Semi-supervised vs the best supervised model, paired by seed** (same hidden labels):

| Labeled | Pseudo-Labeling | Self-Training | Label Spreading |
|---|---|---|---|
| 1 % | −0.046 (won 0/5 seeds) | −0.478 (0/5) | −0.077 (0/5) |
| 5 % | −0.008 (0/5) | −0.031 (0/5) | −0.067 (0/5) |
| 10 % | −0.000 (2/5) | −0.023 (0/5) | −0.061 (0/5) |
| 20 % | −0.000 (2/5) | −0.009 (0/5) | −0.054 (0/5) |

**Why semi-supervised learning did not help here** (from the pseudo-label analysis):

- **The supervised baseline is already strong.** With only 604 labeled reviews (1 %), a linear
  model on TF-IDF reaches 0.819; with all 60k labels it reaches 0.862. Short food-delivery
  reviews carry very clear words ("عالی", "سرد", "دیر"), so unlabeled data adds little.
- **Self-Training collapses at 1 % (confirmation bias).** The accuracy of its pseudo-labels falls
  from 99 % in iteration 1 to about 11 % in iteration 4; it ends up predicting "positive"
  for 99 % of negative reviews.
- **Pseudo-Labeling** adds pseudo-labels that are 93 % correct at 1 %, but those ~400 wrong
  labels outweigh many of the 604 true ones; the loss is in positive-class recall
  (0.60 vs 0.77 for Linear SVM). From 10 % on it ties the supervised model.
- **Label Spreading** is weakest everywhere (even at 100 %), because its 300 LSA dimensions
  carry less information than full TF-IDF; its pseudo-labels are only 72–78 % correct.

Full tables: `artifacts/results/default/report.md`; discussion in
[`docs/explanations/phase-5.md`](docs/explanations/phase-5.md) (Persian).

---

## Reproducing the results

Everything is driven by `configs/default.yaml` and the `reviewlens` CLI:

```bash
reviewlens data download        # raw CSVs from Hugging Face (pinned revision) -> data/raw/
reviewlens data prepare         # normalize (hazm), validate, deduplicate -> data/processed/
reviewlens run-experiments      # 7 methods x 5 fractions x 5 seeds, ~20-30 min on a laptop
reviewlens evaluate             # report tables + figures from the saved CSVs, seconds
reviewlens train -m logistic_regression -f 1.0 --save   # model for the app and the agent
```

- Dataset: [ParsiAI/snappfood-sentiment-analysis](https://huggingface.co/datasets/ParsiAI/snappfood-sentiment-analysis);
  its official `test.csv` is the fixed test set.
- No leakage: TF-IDF and LSA are fit on training texts only (labeled + unlabeled).
- `run-experiments` saves after every run; an interrupted experiment continues where it
  stopped (`--fresh` starts over). Re-running everything on a second machine reproduced
  the table above (one cell differed by 0.001, likely from newer library versions).
- Outputs: `artifacts/results/default/` (`runs.csv`, `pseudo_labels.csv`, `summary.csv`,
  `report.md`, ...) and `artifacts/figures/default/`. `data/` and `artifacts/` are never committed.

Other commands: `reviewlens --help`, `reviewlens show-config`,
`reviewlens baselines --fraction 0.05`, `reviewlens -e configs/experiment/quick.yaml run-experiments`
(small smoke test).

---

## The app

`make app` starts a Persian, right-to-left Streamlit app with three pages:

1. **Analyze review:** sentiment, confidence and the most important words of one review,
   straight from the trained model. Needs no language model or API key.
2. **Results:** the answer to the research question, the macro-F1 table and the main figure.
3. **Agent chat:** chat with the agent; every answer shows its tool-call trace
   (step, tool, input, output).

---

## The agent and its three run modes

The language model decides which tool to call, with what input and how many times; the code
only runs the tools and records every call (no if-based routing). Tools are plain, read-only
Python functions:

| Tool | Returns |
|---|---|
| `analyze_review(text)` | sentiment, confidence, top words (from the linear model's coefficients) |
| `get_experiment_results(method?, label_fraction?)` | mean ± std macro-F1, or the best method per fraction |
| `search_reviews(query, sentiment?, limit≤5)` | up to 5 matching reviews with their labels |

Every tool input is validated with pydantic; invalid input, unknown tools and missing files
come back to the model as error objects, never as crashes. The system prompt requires
simple Persian answers based only on tool results, and states that tool results (review
texts in particular) are **data, never instructions** (prompt-injection defence).

| Mode | How |
|---|---|
| **API** (default) | `ANTHROPIC_API_KEY`; optional `ANTHROPIC_MODEL` (default `claude-opus-5-5`) |
| **Local model** (Ollama) | Ollama running, a model with tool calling, `OPENAI_MODEL`, and the override `configs/agent/ollama.yaml` |
| **No model** | the Analyze page and `reviewlens agent local-test` work without any model |

```bash
reviewlens agent local-test                                   # all tools, no model, no key
reviewlens agent chat "این نظر را تحلیل کن: غذا سرد بود"
reviewlens -e configs/agent/ollama.yaml agent chat "..."      # local model mode
reviewlens agent eval                                         # 12 cases x 3 runs -> docs/agent_tests.md
```

For the app in local model mode (PowerShell):

```powershell
$env:OPENAI_MODEL = "qwen2.5:7b"
$env:REVIEWLENS_EXPERIMENT = "configs/agent/ollama.yaml"
make app
```

Keys live only in environment variables (names in `.env.example`); a missing key, an
invalid key or an unreachable server produces a clear Persian message.

> **Status of the agent evaluation:** the agent is tested with a scripted fake model
> (no network) and `agent local-test` passes 12/12 on the real data. The evaluation with a
> real language model (`agent eval`, via Ollama) **has not been run yet**;
> [`docs/agent_tests.md`](docs/agent_tests.md) lists the cases and will hold the results.

---

## Tests and code quality

- **255 tests** (pytest) on tiny fixture data: data pipeline, models, experiment runner,
  evaluation, agent tools, agent loop with a fake model, providers (no network), CLI, and
  all three app pages (Streamlit `AppTest`, headless).
- **98 % line coverage** (`make coverage`). The Streamlit page scripts are tested with
  AppTest but are not counted, because coverage cannot attribute lines of scripts that
  Streamlit runs itself.
- `ruff` (lint + format) and `mypy --strict`: clean. `pre-commit` runs both on every commit
  (activate `.venv` first).

---

## Project layout

```
configs/            default.yaml, experiment/*.yaml, agent/ollama.yaml
src/reviewlens/
  data/             download, load, validate, prepare, split + label masking
  preprocessing/    Persian normalizer (hazm)
  features/         TF-IDF, LSA
  models/           Classifier protocol, registry, supervised, semi-supervised, TextClassifier
  experiments/      one training run; the grid runner with resume
  evaluation/       metrics, report tables, figures
  agent/            tools, schemas, system prompt, providers, loop, evaluation
  app/              Streamlit app (one file per page)
  cli.py            typer entry point (`reviewlens`)
tests/              255 tests + tiny fixture dataset
notebooks/01_eda.ipynb
docs/               design sheet, prompt notebook, agent tests, report outline,
                    Persian phase explanations
```

The full specification is in [`CLAUDE.md`](CLAUDE.md). How AI was used to build the project,
with the important prompts, errors and fixes, is in [`docs/prompts.md`](docs/prompts.md).

---

## Limitations

- One dataset (short food-delivery reviews) and one feature type (TF-IDF); results may
  differ on longer texts, more classes or weaker baselines.
- Hyperparameters were deliberately not tuned (tuning on the test set would leak, and with
  1 % labels there is too little data to tune reliably).
- The agent has not yet been evaluated with a real language model (see above).
