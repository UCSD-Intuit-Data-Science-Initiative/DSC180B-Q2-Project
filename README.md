# Workforce Optimization for Intuit's Virtual Expert Platform
Static Informational Website: [https://ucsd-intuit-data-science-initiative.github.io/DSC180B-Q2-Project/](https://ucsd-intuit-data-science-initiative.github.io/DSC180B-Q2-Project/)

## Introduction

Intuit, a leading financial software company, offers products like TurboTax and QuickBooks, and connects users with live experts through its Virtual Expert Platform (VEP). This platform’s operational efficiency relies heavily on accurate demand forecasting and precise supply planning. By effectively matching expert availability with user needs, Intuit ensures timely support and maintains cost-effectiveness.

Forecasting models in workforce management set the stage for everything else. The expectation is to accurately capture when, how much, and what kind of demand will show up. For example, it’s not enough to know that thousands of customers will seek help next week — the models must account for hourly patterns, seasonal spikes, and even unexpected events. Accuracy matters because small errors in forecasts ripple downstream, leading to overstaffing (higher costs) or understaffing (longer wait times and lower customer satisfaction).

On the supply planning side, the challenge is about optimization under constraints. Supply planning requires building staffing schedules that balance multiple, often conflicting requirements, such as ensuring the right mix of skills (e.g., tax experts vs. bookkeeping experts), covering peak hours without burning out the team, staying within budget and labor regulations, and allowing flexibility for last-minute changes when forecasts shift.day-to-day.

## Getting Started

Only [Docker Desktop](https://www.docker.com/products/docker-desktop/) is required to run the full stack.

```bash
git clone <repo>
cd <repo>
make backend-up     # start FastAPI backend (trains ML model on startup)
make frontend-up    # start React dashboard
```

Then open:
- **Dashboard:** http://localhost:3000
- **API docs:** http://localhost:8000/docs

```bash
make install        # installs dependencies & pre-commit hooks (requires Poetry 1.8+)
make lint           # sanity-check tooling
make test           # run the sample unit tests
make pipeline-run   # execute the demo training pipeline (no Docker)
```

Poetry 1.8+ is required for local development. If Poetry is not already installed, follow the [official instructions](https://python-poetry.org/docs/#installation).

## Repository Layout

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

## Tooling Highlights

- **Poetry** – dependency and environment management (`make install`).
- **Ruff** – linting & formatting (`make lint`, `make format`).
- **mypy** – static type checks to catch bugs early.
- **Tox** – orchestrates lint/type/test/coverage in isolated envs (`poetry run tox`).
- **Pytest** – unit test runner with coverage configured.
- **Loguru** – opinionated logging with contextual helpers.
- **Pre-commit** – consistent formatting & linting before commits.
- **GitHub Pages ready docs** – `docs/` contains the scaffolding for project documentation.
- **Streamlit** - Python framework for interactive data app hosting.
