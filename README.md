# Python Data Science Template

Welcome! This repository is a ready-to-use scaffold for UCSD Intuit Data Science Capstone projects. The goal is to minimize setup friction so you can concentrate on experimentation, modelling, and storytelling. This document walks you through how the template is organised and how to work with it day-to-day.

## Table of Contents

1. [Quickstart](#1-quickstart)
2. [Repository Layout](#2-repository-layout)
3. [Tooling Highlights](#3-tooling-highlights)
4. [Common Commands](#4-common-commands)
5. [Makefile Cheatsheet](#5-makefile-cheatsheet)
6. [Getting Started (Detailed)](#5-getting-started-detailed)
7. [Working with the Package (`src/main_module`)](#6-working-with-the-package-srcmain_module)
8. [Example Usage](#7-example-usage)
9. [Data & Notebook Practices](#8-data--notebook-practices)
10. [Pipelines & Scripts](#9-pipelines--scripts)
11. [Testing Strategy](#10-testing-strategy)
12. [Linting, Formatting & Pre-commit Hooks](#11-linting-formatting--pre-commit-hooks)
13. [Documentation Workflow](#12-documentation-workflow)
14. [Logging](#13-logging)
15. [Best Practices & Next Steps](#14-best-practices--next-steps)
16. [Need Help?](#15-need-help)

## 1. Quickstart

```bash
git clone <repo>
cd <repo>
make install        # installs dependencies & pre-commit hooks
make lint           # sanity-check tooling
make test           # run the sample unit tests
make pipeline-run   # execute the demo training pipeline
```

Poetry 1.8+ is required. If Poetry is not already installed, follow the [official instructions](https://python-poetry.org/docs/#installation).

## 2. Repository Layout

```
├── data/                 # Local datasets (kept out of Git)
│   ├── raw/              # Immutable source data
│   ├── processed/        # Cleaned / feature engineered tables
│   ├── interim/          # Scratch layers or temporary outputs
│   └── external/         # Third-party, licensed, or public data
├── notebooks/            # Exploratory analysis (Jupyter, etc.)
├── models/               # Serialized models / experiment artefacts
├── references/           # Datasheets, dictionaries, design docs
├── reports/              # Generated reports, `figures/` for visuals
├── scripts/              # CLI utilities & pipelines (see demo script)
├── src/
│   └── main_module/      # Installable Python package
│       ├── data/         # Loading + preprocessing helpers
│       ├── modeling/     # Training, evaluation, prediction utilities
│       ├── utils/        # Reusable helpers (I/O, time, logging)
│       ├── visualization/# Plotting utilities
│       └── logging.py    # Loguru configuration & helpers
├── tests/
│   ├── unit/             # Pytest unit tests
│   └── integration/      # Placeholder for future integration suites
├── docs/                 # GitHub Pages-ready documentation
├── Makefile              # High-level automation entrypoints
├── pyproject.toml        # Poetry + tool configuration
├── tox.ini               # Tox environments for lint/type/tests
└── README.md             # You are here
```

> Pro tip: folders that should stay in Git even when empty have `.gitkeep` files.

## 3. Tooling Highlights

- **Poetry** – dependency and environment management (`make install`).
- **Ruff** – linting & formatting (`make lint`, `make format`).
- **mypy** – static type checks to catch bugs early.
- **Tox** – orchestrates lint/type/test/coverage in isolated envs (`poetry run tox`).
- **Pytest** – unit test runner with coverage configured.
- **Loguru** – opinionated logging with contextual helpers.
- **Pre-commit** – consistent formatting & linting before commits.
- **GitHub Pages ready docs** – `docs/` contains the scaffolding for project documentation.

## 4. Common Commands

| Task | Command | What it does |
| --- | --- | --- |
| Install project | `make install` | Creates Poetry env, installs deps, sets up pre-commit |
| Format code | `make format` | Ruff auto-fix (format + lint) then cleans caches |
| Lint & type-check | `make lint` | Ruff linting and mypy typing |
| Run unit tests | `make test` | Executes tox `py` env (pytest + coverage) |
| Run full tox suite | `poetry run tox` | Lint + type + unit tests in one go |
| Pre-commit hooks | `make precommit` | Runs all hooks across the repo |
| Demo pipeline | `make pipeline-run` | Executes `scripts/run_pipeline.py --demo` |
| Pipeline help | `make pipeline-help` | Shows CLI options for pipeline script |
| Clean caches | `make clean` | Removes tox caches, coverage files, build artefacts |
| Serve docs stack | `make docs-up` | Builds containers and serves docs (Podman if available) |
| Stop docs stack | `make docs-down` | Stops containers and removes orphans |
| Inspect docs status | `make docs-ps` | Shows running docs services |
| Tail docs logs | `make docs-logs` | Streams Jekyll container logs |
| Build docs static site | `make docs-build` | Runs Jekyll build inside container |
| Check doc links | `make docs-check` | Runs lychee link checker in container |
| Open docs in browser | `make docs-open` | Launches `http://localhost:4000` locally |

## 5. Makefile Cheatsheet

The `Makefile` is your command dashboard. Every target includes a description surfaced via `make help`.

### Setup
- `make install` – install dependencies and activate pre-commit hooks.

### Quality
- `make lint` – Ruff lint + mypy type-check.
- `make format` – Ruff auto-fix (lint + formatter) followed by cache cleanup.
- `make test` – runs tox `py` environment (pytest with coverage).

### Pipelines
- `make pipeline-run` – runs the demo pipeline (`scripts/run_pipeline.py --demo`).
- `make pipeline-help` – displays CLI usage for the pipeline script.

### Documentation
- `make docs-up` – start the containerised docs stack (Podman preferred, Docker fallback).
- `make docs-down` – stop containers and clean up.
- `make docs-build` – perform a one-off Jekyll build inside the container.
- `make docs-check` – run lychee link checking against the generated site.
- `make docs-open` – open the served docs in your default browser.

### Maintenance
- `make clean` – orchestrates clean-pyc, clean-build, clean-test, clean-cache.
- `make precommit` – runs all pre-commit hooks on the codebase.

> Tip: `make help` prints colourised sections so newcomers can discover commands quickly.

## 5. Getting Started (Detailed)

1. **Bootstrap the environment**
   - Install Poetry; ensure Python 3.11+ is available (template targets 3.11).
   - Clone the repository and run `make install` to create the virtualenv and install dependencies.
2. **Explore the code base**
   - Start in `src/main_module/` to understand the package structure.
   - Review the sample tests in `tests/unit/` to see how utilities are expected to behave.
3. **Run baseline checks**
   - `make lint` ensures coding standards.
   - `make test` executes pytest with coverage reports in `htmlcov/` and `coverage.xml`.
4. **Execute the demo pipeline**
   - `make pipeline-run` invokes a sample training script that uses synthetic data via the `--demo` flag. Use it as a blueprint for future pipelines.
5. **Publish documentation (optional)**
   - `docs/` is ready for GitHub Pages. Add markdown pages, update navigation, and enable Pages in GitHub settings.

## 6. Working with the Package (`src/main_module`)

- `data/` – functions such as `load_csv`, `clean_dataframe`, and `select_columns` handle ingestion and basic cleaning.
- `modeling/` – contains reusable components (`train_linear_model`, `run_predictions`, `mean_squared_error`) and data classes (`LinearModelResult`).
- `utils/` – generic helpers (`ensure_directory`, `timestamp`) and logging configuration.
- `visualization/` – plotting extensions (currently histogram helper) ready to grow with project needs.
- `logging.py` – central Loguru setup; use `from main_module.logging import get_logger` to obtain a contextual logger in any module.

Each module ships with type hints and docstrings using the provided template. Extend them by following the same style for consistency.

## 7. Example Usage

### Train from a Python session

```python
from pathlib import Path

import pandas as pd

from main_module.data import clean_dataframe, load_csv
from main_module.logging import get_logger
from main_module.modeling import train_linear_model

logger = get_logger("example")

df = load_csv(Path("data/raw/example.csv"))
features = clean_dataframe(df.drop(columns=["target"]))
target = df["target"]
result = train_linear_model(features, target)

logger.info("Intercept", value=result.intercept)
logger.info("Coefficients", values=result.coefficients.tolist())
```

### Run the pipeline on real data

```bash
poetry run python scripts/run_pipeline.py \
  --data data/raw/train.csv \
  --target sale_price
```

### Execute a focused test module

```bash
pytest tests/unit/test_modeling.py -k linear
```

## 8. Data & Notebook Practices

- Keep raw inputs in `data/raw/` and never mutate them.
- Use `data/interim/` for scratch outputs; `data/processed/` for cleaned datasets ready for modelling.
- The repo ignores these directories by default; `.gitkeep` keeps the structure visible.
- Place exploratory notebooks in `notebooks/` and convert reusable logic into `src/main_module/` modules so it’s testable and shareable.

## 9. Pipelines & Scripts

- `scripts/run_pipeline.py` demonstrates a CLI-driven workflow:
  - Accepts `--data` and `--target` arguments for real datasets.
  - Supports `--demo` to generate synthetic data so the pipeline runs without external files.
  - Logs progress (loading data, training, coefficients) via Loguru.
- Add additional CLI utilities to `scripts/`; the Makefile includes helper targets so teammates can run them without remembering long commands.

## 10. Testing Strategy

- Unit tests live under `tests/unit/`. Add new tests that mirror the structure of `src/main_module/`.
- `tests/integration/` is available for end-to-end or pipeline-level tests when you need them.
- Tox ensures tests run inside isolated envs so local development matches CI behaviour.

## 11. Linting, Formatting & Pre-commit Hooks

- Linting uses Ruff (PEP8 + extras) and mypy for typing. When adding new modules, run `make lint` before committing.
- Formatting also uses Ruff; no separate Black/isort configuration is required (and Ruff enforces import style automatically).
- `.pre-commit-config.yaml` runs Ruff, autoflake, mypy, and whitespace checks. Install hooks once via `make install`, then they’ll run automatically on `git commit`.

## 12. Documentation Workflow

**Overview**
- The docs are a Jekyll site using the `just-the-docs/just-the-docs` remote theme.
- Configuration lives in `docs/_config.yml`; markdown content is under `docs/*.md`.
- The Makefile targets wrap `podman-compose`/`docker compose` so you rarely need raw container commands.

**Daily editing loop**
1. Update or add markdown pages in `docs/`.
2. Adjust front matter (`title`, `nav_order`, `layout`) to control sidebar/nav placement.
3. Commit changes along with any supporting assets.

**Build without serving**
- `make docs-build` → runs `bundle exec jekyll build` inside the container, populating `docs/_site/`.
- Use when you only need a static build (e.g., CI, link checking) or to confirm the site compiles.

**Serve locally (recommended)**
- `make docs-up` → builds the images (if needed) and starts Jekyll + LiveReload.
  - Podman is used automatically when present; otherwise Docker Desktop is used.
  - Logs stream through the Makefile; inspect them via `make docs-logs`.
- Optional helpers while the stack is running:
  - `make docs-open` → opens `http://localhost:4000` in your default browser.
  - `make docs-ps` → shows container status (handy for troubleshooting).
- To stop everything, run `make docs-down` (also clears orphan containers).

**Link checking**
- `make docs-check` → executes the lychee link checker in the container against the built site.
- Review warnings for missing anchors or external resources; adjust `.lycheeignore` when needed.

**Styling and theme tweaks**
- Update site metadata, header links, or footer text in `docs/_config.yml`.
- For deeper theme overrides, add Sass or JS under `docs/assets/` following Just the Docs override conventions.
- The upstream theme emits Sass deprecation warnings; they are harmless but documented so you can track upstream changes.

**Publishing**
- Push changes to GitHub and set the repository’s Pages source to `/docs` for automatic deployment.
- Confirm the build in GitHub Pages settings; the Just the Docs theme loads remotely so no extra assets are needed.

## 13. Logging

- Loguru is set up project-wide. Call `get_logger("some.module")` to create a contextual logger.
- Default format includes timestamp, level, module, and message. Adjust `DEFAULT_FORMAT` or add handlers in `configure_logging` if needed (e.g., JSON logs, file output).

## 14. Best Practices & Next Steps

- **Iterate in notebooks, productionalise in `src/`**: keep business logic in the package so it can be tested and reused.
- **Keep tests close to the code**: whenever you add or modify a module, update the matching tests under `tests/`.
- **Run lint/tests before pushing**: `make lint` and `make test` catch issues early. Use `poetry run tox` for a clean-room validation.
- **Document as you go**: update README sections and `docs/` whenever you add new components, pipelines, or data sources.
- **Version data artefacts thoughtfully**: large files should live outside Git; consider DVC, cloud buckets, or shared drives if the project evolves.
- **Set up CI/CD**: future enhancements include GitHub Actions for lint/test and optionally Docker images for reproducible deployment.

## 15. Need Help?

- Inspect `Makefile` to discover additional commands.
- Review `tox.ini` and `pyproject.toml` for configuration details.
- Reach out to teammates or course staff if tooling fails—share command output for quicker diagnosis.

Happy building! This template should get you from a blank repo to a collaborative, reproducible data science project with minimal friction. Iterate, document, and adapt it to your team’s workflow.
