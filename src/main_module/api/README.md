# Workforce API — Backend

FastAPI backend that runs the ML pipeline and serves results to the React dashboard. Built with FastAPI + Uvicorn + scikit-learn.

Runs at **`localhost:8000`** with hot reload.

## Prerequisites

Only [Docker Desktop](https://www.docker.com/products/docker-desktop/) is required — no Python or pip needed locally.

## Running the backend

From the **project root**:

```bash
make backend-up     # build image + start FastAPI server
make backend-open   # open localhost:8000/docs in browser
make backend-logs   # tail server output / errors
make backend-down   # stop the container
```

Hot reload is enabled — edits to any `.py` file restart the server automatically without rebuilding.

## Endpoints

| URL | Description |
|---|---|
| `localhost:8000` | Health check |
| `localhost:8000/api/metrics?date=2025-04-15` | Day-level summary (total calls, peak agents, avg SLA, avg wait) |
| `localhost:8000/api/forecast?date=2025-04-15` | Predicted call volume per 30-min slot |
| `localhost:8000/api/staffing?date=2025-04-15` | Staffing schedule per 30-min slot |
| `localhost:8000/docs` | Interactive Swagger UI to test all endpoints |

The `/api/staffing` endpoint also accepts constraint parameters matching the React simulation sliders:

```
/api/staffing?date=2025-04-15&min_sla=0.90&max_wait=45&max_occupancy=0.80
```

## Project structure

```
api/
├── main.py          # FastAPI app — endpoints + ML pipeline logic
├── requirements.txt # Python dependencies
├── Dockerfile       # dev + production Docker stages
└── README.md        # you are here
```

## How the model loads

The server trains the `HybridForecaster` once at startup (~1 min), then reuses it for every request. If the data file is missing, it returns placeholder data instead of crashing.

The data file must be at `data/interim/mock_intuit_2year_data.csv` from the project root.

## Tech stack

| Tool | Version | Purpose |
|---|---|---|
| FastAPI | 0.115+ | Web framework |
| Uvicorn | 0.30+ | ASGI server |
| scikit-learn | 1.5+ | ML models |
| XGBoost / LightGBM | 2.0+ / 4.0+ | Boosting models (optional) |
| Optuna | 3.6+ | Hyperparameter tuning |
| Docker | any | Containerized dev environment |

## Docker details

The `Dockerfile` has two stages:

- **`dev`** (used by `make backend-up`) — runs Uvicorn with hot reload
- **`production`** — runs Uvicorn without hot reload, copies source files into the image

The `docker-compose.yml` at the project root mounts your local files into the container via a volume, which is what enables hot reload without rebuilding the image.