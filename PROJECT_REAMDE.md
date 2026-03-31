# Workforce Optimization for Intuit's Virtual Expert Platform

A comprehensive call center workforce management system built with ML-driven demand forecasting and Erlang-A queueing optimization. This project demonstrates how to build production-grade staffing systems for customer support platforms.

## Features

- **ML-Driven Demand Forecasting** — Random Forest + Gradient Boosting ensemble with 15 features (cyclical time, tax seasonality, holidays, lag features). R² = 0.974, WMAPE = 11.34%
- **Erlang-A Queueing Optimization** — Binary search + vectorized queue theory calculations for accurate staffing. Response time: 0.2 seconds per day
- **Per-Slot AHT Lookup** — 336 real average handle times (day-of-week × 30-min slot) from 32M historical calls. Realistic range: 660s–1340s
- **Fast API Backend** — Pre-trained model bundle loaded once via `@lru_cache`. Startup time: <2 seconds
- **Interactive React Dashboard** — Real-time KPI cards, demand forecasts, staffing schedules, and weekly trends
- **Docker-Based Deployment** — One-command setup for full stack (backend + frontend)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI, scikit-learn (RF, GB), NumPy (vectorized math) |
| **Frontend** | React, TypeScript, Recharts |
| **ML Training** | Python, pandas, scikit-learn |
| **Queueing Theory** | Erlang-A formula (call center analytics) |
| **Infrastructure** | Docker, Docker Compose |

## Getting Started

Only [Docker Desktop](https://www.docker.com/products/docker-desktop/) is required.

```bash
# Clone and enter directory
git clone <repository-url>
cd DSC180B-Q2-Project

# Start backend (FastAPI + pre-trained model)
make backend-up

# Start frontend (React dashboard)
make frontend-up
```

Then open:
- **Dashboard:** http://localhost:3000
- **API Docs:** http://localhost:8000/docs (Swagger interactive docs)

## Architecture Overview

### High-Level Flow

```
Historical Call Data (32M records)
  ↓ [Offline: scripts/train_model.py]
  ├→ Feature Engineering (15 features)
  ├→ Random Forest (500 trees)
  ├→ Gradient Boosting (1000 trees)
  ├→ AHT Lookup (336 day/slot combinations)
  └→ Pickle Bundle (~1.5 GB)
  ↓ [API Startup: @lru_cache loads bundle once]
  ↓ [Per Request: <0.2 seconds]
  ├→ Forecast Call Volume (50/50 RF+GB average)
  ├→ Look Up Slot-Specific AHT
  ├→ Run Erlang-A Optimizer (binary search)
  └→ Generate Staffing Schedule
  ↓ [React Dashboard]
  └→ Display KPIs + Charts + Tables
```

### Core Components

#### 1. ML Forecaster (`src/main_module/modeling/`)
- **Models:** Random Forest (500 trees, log2 features) + Gradient Boosting (1000 trees, learning_rate=0.01)
- **Features (15 total):**
  - Cyclical time: `hour_sin`, `hour_cos`, `day_sin`, `day_cos`
  - Calendar: `month`, `weekofyear`, `is_january`
  - Holidays: `is_minor_holiday`, `is_major_holiday`
  - Tax season: `days_to_tax_day`, `is_tax_season`, `is_post_tax_drop`
  - Memory: `lag_1weeks`, `trend_1w`, `max_1w` (historical context)
- **Ensemble Strategy:** 50/50 average of RF and GB predictions
- **Business Logic:** Major holidays override ML predictions with historical profile averages
- **Model Card:** See `docs/model_card_forecasting_dynamic_weeks.md` for detailed metrics

#### 2. Erlang-A Optimizer (`src/main_module/workforce/`)
- **CallCenterEmulator:** Simulates queue dynamics (arrivals, handle times, caller patience, abandonment)
- **SupplyOptimizer:** Uses binary search (O(log n)) to find minimum agents meeting SLA constraints
- **Erlang-B Formula:** Vectorized with NumPy (O(n) instead of O(n²))
- **Constraints:** Min SLA compliance %, max wait time, max agent utilization

#### 3. FastAPI Backend (`src/main_module/api/main.py`)

| Endpoint | Purpose | Example Response |
|----------|---------|---------|
| `GET /` | Health check | `{"status": "ok", "model_ready": true}` |
| `GET /api/metrics?date=YYYY-MM-DD` | Day-level KPI summary | Total calls, peak agents, avg SLA%, avg wait time |
| `GET /api/forecast?date=YYYY-MM-DD` | 30-min demand forecast | Array of {time, predicted_calls, model_used} |
| `GET /api/staffing?date=YYYY-MM-DD` | Full staffing schedule | Array of {time, agents, SLA%, wait_time, utilization, AHT, feasible} |
| `GET /api/weekly-forecast?week_start=YYYY-MM-DD` | 7-day trend forecast | Array of {date, total_calls, error_range} |

#### 4. React Dashboard (`src/main_module/visualization/`)
- **KPI Cards:** SLA compliance, average wait time, agent occupancy, total calls processed
- **Demand Chart:** 30-min slot granularity with area visualization
- **Staffing Table:** Real-time agents needed, SLA%, wait times per slot
- **Weekly Forecast:** 7-day bar chart with error bars, current week highlighted

## Key Improvements Made

### Performance Optimization
| Change | Before | After | Impact |
|--------|--------|-------|--------|
| SupplyOptimizer search | Linear (O(n)) | Binary search (O(log n)) | 8.5 min → 0.2 sec |
| Erlang-B formula | O(n²) Python loop | O(n) NumPy vectorized | 95% faster |

### Model Quality
- **15 Features:** From 7 basic calendar features to comprehensive lag + seasonality
- **Ensemble:** 50/50 RF + GB (captures non-linear patterns better than single model)
- **Validation:** R² = 0.974, WMAPE = 11.34% (on normal business days)

### Realistic Staffing
- **Per-Slot AHT:** Real handle times (660s–1340s) vs. hardcoded 600s everywhere
- **336 Entries:** Day-of-week × 30-min slot combinations
- **Zero Startup Cost:** Pre-computed offline, bundled in pkl

### Data Handling
- **Outlier Smoothing:** 2025-08-29 data corruption handled via time-interpolation
- **Lag Feature Fallback:** Exact historical match → nearest-neighbor when out-of-range
- **Answered-Only:** Only 31.8M answered calls included in AHT calculation (excluded abandoned)

## Repository Layout

```
├── data/
│   ├── parquet/               # Raw call data (32M records, 2+ years)
│   ├── models/                # Pre-trained pkl bundle (1.5 GB)
│   ├── interim/               # Temporary cache files
│   └── external/              # Reference data
├── notebooks/
│   └── pipeline_validation.ipynb   # End-to-end validation notebook
├── scripts/
│   └── train_model.py         # Offline RF+GB training + AHT lookup
├── src/main_module/
│   ├── api/                   # FastAPI backend (main.py + requirements)
│   ├── modeling/              # RF + GB ensemble training
│   ├── workforce/             # Erlang-A emulator + optimizer
│   └── visualization/         # React dashboard (TypeScript + Recharts)
├── docs/                      # GitHub Pages source + model cards
├── Makefile                   # Docker orchestration targets
├── docker-compose.yml         # App stack (backend + frontend)
├── docker-compose.docs.yml    # Docs stack (Jekyll, optional)
└── README.md                  # You are here
```

## Running Offline Training

Pre-compute the RF+GB ensemble and AHT lookup:

```bash
# Requires: Python 3.9+, scikit-learn, pandas, numpy, pyarrow
PYTHONPATH=src python scripts/train_model.py

# Output: data/models/call_volume_model_bundle.pkl (~1.5 GB)
# Contains:
#   - rf_model: Random Forest (500 trees)
#   - gb_model: Gradient Boosting (1000 trees)
#   - features: [15 feature names]
#   - holiday_profiles: {(month, day, time): mean_calls}
#   - daily_std_lookup: {day_of_week: std_dev}
#   - aht_lookup: {(day_of_week, "HH:MM"): mean_aht_seconds}
#   - call_volume_history: pd.Series (34K slots, 300KB)
#   - forecast_weeks: 1
#   - trained_at: ISO timestamp
```

Training time: ~2 minutes on modern hardware (parallel RF + GB).

## Development

Local development requires [Poetry](https://python-poetry.org/docs/#installation) 1.8+.

```bash
# Install dependencies + pre-commit hooks
make install

# Code quality
make lint           # Format + type-check
make format         # Auto-format with Black/isort
make test           # Run pytest suite

# Local pipeline (no Docker)
make pipeline-run   # Train model offline
```

## Documentation (Optional)

The `docs/` directory contains GitHub Pages source with Jekyll.
Docs tooling runs in a **separate** Docker Compose stack (not required for dashboard/API).

```bash
make docs-up        # Start Jekyll server at http://localhost:4000
make docs-check     # Run link checker (Lychee)
make docs-down      # Stop docs containers
```

Commands use `docker-compose.docs.yml` and do not affect backend/frontend containers.

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Model Training | ~2 minutes (offline, parallel) |
| API Startup | <2 seconds (@lru_cache + emulator init) |
| Per-Request Latency | 0.2 seconds (24 slots × binary search) |
| Model Size | 1.5 GB (sklearn objects + history) |
| AHT Lookup | 336 entries, O(1) dict lookup |

## Deployment Notes

### Docker Stack
- **Backend**: FastAPI with uvicorn, auto-reload on code changes
- **Frontend**: React with Vite dev server
- **Model Loading**: Pre-trained pkl loaded once on startup
- **Volume Mounts**: `data/models/` excluded from auto-reload to prevent watch loops

### Production Considerations
- Pre-train model offline and mount pkl as read-only volume
- Use `--reload-exclude data/*` in Uvicorn to prevent reload loops from cache writes
- Scale horizontally with multiple API instances (stateless)
- Model bundle size (1.5 GB) may require optimization for edge deployments

## Future Work

- Multi-day forecasts with confidence intervals
- Skill-based routing (tax experts vs. general support)
- Real-time anomaly detection + dynamic reforecasting
- Cost optimization (salary vs. SLA trade-offs)
- A/B testing framework for staffing strategies

## References

- **Model Card:** `docs/model_card_forecasting_dynamic_weeks.md`
- **API Docs:** `http://localhost:8000/docs` (when backend running)
- **Dashboard:** `http://localhost:3000` (when frontend running)
- **Workflow:** See `notebooks/pipeline_validation.ipynb` for end-to-end validation

## Requirements

- Docker Desktop (for containerized stack)
- Python 3.9+ (for local development)
- Poetry 1.8+ (for dependency management)
- 2GB disk space (for model bundle + parquet data)

## License

See LICENSE file for details.

## Contributing

1. Create a feature branch from `master`
2. Make changes and test locally (`make test`)
3. Format code (`make format`, `make lint`)
4. Commit with clear messages
5. Submit PR with summary of changes

## Acknowledgments

Built with [FastAPI](https://fastapi.tiangolo.com/), [scikit-learn](https://scikit-learn.org/),
and [React](https://react.dev/). Queue theory based on Erlang-A model for call centers.
