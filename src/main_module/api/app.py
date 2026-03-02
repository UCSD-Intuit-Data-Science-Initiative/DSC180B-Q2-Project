"""
app.py
------
Lean FastAPI server that loads a pre-trained model from forecaster.joblib
and serves a single /predict POST endpoint.

Prerequisite: run train.py first to generate forecaster.joblib.

Usage:
    PYTHONPATH=src uvicorn src.main_module.api.app:app --port 8001 --reload
"""

import sys
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MODEL_PATH = Path(__file__).resolve().parent / "forecaster.joblib"

# ---------------------------------------------------------------------------
# Load model at startup (module level — no training, just a file read)
# ---------------------------------------------------------------------------

if not _MODEL_PATH.exists():
    print("=" * 60, file=sys.stderr)
    print(f"ERROR: {_MODEL_PATH} not found.", file=sys.stderr)
    print("Run train.py first to generate the model file:", file=sys.stderr)
    print("  PYTHONPATH=src python src/main_module/api/train.py", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    sys.exit(1)

print(f"Loading model from {_MODEL_PATH}...")
forecaster = joblib.load(_MODEL_PATH)
print("Model loaded.")

from main_module.workforce import (
    CallCenterEmulator,
    EmulatorConfig,
    OptimizationConstraints,
    SupplyOptimizer,
)

emulator = CallCenterEmulator(
    config=EmulatorConfig(
        avg_handle_time=300,
        sla_threshold_seconds=60,
        interval_duration_seconds=1800,
    )
)
optimizer = SupplyOptimizer(emulator, max_supply=500)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Workforce Optimization API (app.py)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    date: str
    min_sla: float = 0.80
    max_wait: float = 60.0
    max_occupancy: float = 0.85

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "app.py is running — model loaded from forecaster.joblib"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Forecast demand and calculate required staffing for a given date.

    Body:
        date         : "YYYY-MM-DD"
        min_sla      : minimum SLA as fraction (default 0.80)
        max_wait     : max wait time in seconds (default 60.0)
        max_occupancy: max agent occupancy as fraction (default 0.85)

    Returns a list of 30-min slots with forecast and staffing numbers.
    """
    try:
        demand_df = forecaster.predict_day(req.date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {e}")

    demand_df["time"] = demand_df["interval_start"].dt.strftime("%H:%M")
    demand_df["is_open"] = (
        (demand_df["interval_start"].dt.hour >= 5)
        & (demand_df["interval_start"].dt.hour < 17)
    )
    open_slots = demand_df[demand_df["is_open"] & (demand_df["predicted_calls"] > 0)]

    constraints = OptimizationConstraints(
        min_sla=req.min_sla,
        max_wait_time=req.max_wait,
        max_occupancy=req.max_occupancy,
    )

    results = []
    for _, row in open_slots.iterrows():
        demand = int(row["predicted_calls"])
        supply_result = optimizer.optimize(demand, constraints)
        metrics = supply_result.predicted_metrics
        results.append({
            "time": row["time"],
            "predicted_calls": demand,
            "agents": supply_result.headcount,
            "avg_wait_time": round(metrics.avg_wait_time, 1),
            "sla_compliance": round(metrics.sla_compliance, 1),
            "utilization_rate": round(metrics.utilization_rate, 1),
            "abandonment_rate": round(metrics.abandonment_rate, 1),
            "is_feasible": supply_result.is_feasible,
        })

    return results
