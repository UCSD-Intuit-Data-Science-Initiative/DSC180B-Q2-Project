# Workforce Optimization for Intuit’s Virtual Expert Platform

Website: [https://ucsd-intuit-data-science-initiative.github.io/DSC180B-Q2-Project/](https://ucsd-intuit-data-science-initiative.github.io/DSC180B-Q2-Project/)

## Introduction

Intuit, a leading financial software company, offers products like TurboTax and QuickBooks, and connects users with live experts through its Virtual Expert Platform (VEP). This platform’s operational efficiency relies heavily on accurate demand forecasting and precise supply planning. By effectively matching expert availability with user needs, Intuit ensures timely support and maintains cost-effectiveness.

Forecasting models in workforce management set the stage for everything else. The expectation is to accurately capture when, how much, and what kind of demand will show up. For example, it’s not enough to know that thousands of customers will seek help next week — the models must account for hourly patterns, seasonal spikes, and even unexpected events. Accuracy matters because small errors in forecasts ripple downstream, leading to overstaffing (higher costs) or understaffing (longer wait times and lower customer satisfaction).

On the supply planning side, the challenge is about optimization under constraints. Supply planning requires building staffing schedules that balance multiple, often conflicting requirements, such as ensuring the right mix of skills (e.g., tax experts vs. bookkeeping experts), covering peak hours without burning out the team, staying within budget and labor regulations, and allowing flexibility for last-minute changes when forecasts shift day-to-day.

## Quick Links

| Component | Details |
|---|---|
| [Backend API](src/main_module/api/README.md) | FastAPI endpoints, model loading, Docker config |
| [Frontend Dashboard](src/main_module/visualization/README.md) | React + Vite + Recharts, component structure, tech stack |

## Getting Started

### Prerequisites

- **Python 3.10+** with [Poetry 1.8+](https://python-poetry.org/docs/#installation)
- **Node.js 18+** (for frontend)
- **Docker Desktop** (if using Docker)
- **Data files** (not tracked in git):
  - `data/parquet/dataset_1_call_related.parquet`
  - `data/raw/dataset_2_expert_metadata.parquet`
  - `data/raw/dataset_4_expert_state_interval.parquet`

### Setup

```bash
git clone <repo> && cd <repo>
make install                                    # Python deps + pre-commit hooks
cd src/main_module/visualization && npm install --legacy-peer-deps && cd ../../..
```

### Train the model (required before first run)

```bash
PYTHONPATH=src poetry run python scripts/train_model.py
```

This produces `data/models/call_volume_model_bundle.pkl` which the backend loads at startup. Without it the API returns placeholder data.

### Run with Docker

```bash
make backend-up     # FastAPI on port 8000
make frontend-up    # React dashboard on port 3000
```

### Run locally (no Docker)

```bash
# Terminal 1
make backend-local   # FastAPI on port 8000

# Terminal 2
make frontend-local  # Vite dev server on port 3000
```

Open http://localhost:3000 (dashboard) or http://localhost:8000/docs (API docs).

## Repository Layout

```
├── data/                        # Local data (gitignored)
│   ├── raw/                     # Source parquet files
│   ├── parquet/                 # Call-related dataset
│   └── models/                  # Trained model bundle (.pkl)
├── scripts/
│   ├── train_model.py           # Offline model training
│   └── run_pipeline.py          # Demo training pipeline
├── src/main_module/
│   ├── api/                     # FastAPI backend (see api/README.md)
│   ├── workforce/               # Core ML & scheduling logic
│   │   ├── combined_forecaster.py   # LightGBM demand forecaster
│   │   ├── call_center_emulator.py  # Erlang-C staffing calculator
│   │   ├── supply_optimizer.py      # Supply recommendation engine
│   │   ├── shift_scheduler.py       # Agent shift assignment
│   │   └── agent_analytics.py       # Agent performance metrics
│   └── visualization/           # React frontend (see visualization/README.md)
├── tests/                       # Pytest unit tests
├── docs/                        # GitHub Pages site
├── Makefile                     # All automation targets
├── docker-compose.yml           # Backend + frontend containers
└── pyproject.toml               # Poetry config
```

## Key Make Targets

| Target | Description |
|---|---|
| `make install` | Install Python deps + pre-commit hooks |
| `make backend-up` / `make backend-local` | Start backend (Docker / local) |
| `make frontend-up` / `make frontend-local` | Start frontend (Docker / local) |
| `make backend-logs` / `make frontend-logs` | Tail container logs |
| `make lint` | Ruff + mypy checks |
| `make format` | Auto-format with Ruff |
| `make test` | Run tests via tox |
| `make clean` | Remove caches and build artifacts |
| `make docs-up` | Start Jekyll docs server (port 4000) |
