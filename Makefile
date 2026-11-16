# ReviewLens developer commands.
# Works on Linux/macOS and on Windows with GNU make (`winget install ezwinports.make`).

ifeq ($(OS),Windows_NT)
    PYTHON ?= py -3.12
    VENV_PY := .venv\Scripts\python.exe
else
    PYTHON ?= python3
    VENV_PY := .venv/bin/python
endif

.PHONY: help install test coverage lint format typecheck check app clean

help:
	@echo "make install    create .venv and install the package with dev tools"
	@echo "make test       run the test suite"
	@echo "make coverage   run the tests with a coverage report"
	@echo "make lint       ruff lint + format check + mypy"
	@echo "make format     auto-format and auto-fix with ruff"
	@echo "make check      lint + test"
	@echo "make app        start the Streamlit app"
	@echo "make clean      remove caches and build files"

install:
	$(PYTHON) -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -e ".[dev,openai]"
	$(VENV_PY) -m pre_commit install

test:
	$(VENV_PY) -m pytest

coverage:
	$(VENV_PY) -m pytest --cov=reviewlens --cov-report=term

lint:
	$(VENV_PY) -m ruff check src tests
	$(VENV_PY) -m ruff format --check src tests
	$(VENV_PY) -m mypy

format:
	$(VENV_PY) -m ruff check --fix src tests
	$(VENV_PY) -m ruff format src tests

typecheck:
	$(VENV_PY) -m mypy

check: lint test

# Local model mode: set OPENAI_MODEL and REVIEWLENS_EXPERIMENT=configs/agent/ollama.yaml first.
app:
	$(VENV_PY) -m streamlit run src/reviewlens/app/streamlit_app.py

clean:
	$(VENV_PY) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for n in ('.pytest_cache', '.mypy_cache', '.ruff_cache', 'build', 'dist') for p in [pathlib.Path(n)]]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__') if '.venv' not in p.parts]"
