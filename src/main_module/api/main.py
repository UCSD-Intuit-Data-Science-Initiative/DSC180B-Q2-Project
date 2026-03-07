"""
Workforce Optimization API
--------------------------
FastAPI backend that serves ML pipeline results to the React dashboard.

The RF+GB ensemble model is trained offline by `scripts/train_model.py` and
saved to `data/models/call_volume_model_bundle.pkl`.  At startup this API
loads the bundle once via @lru_cache — no training happens here.

Endpoints:
  GET /                          health check - confirms server is running
  GET /api/metrics?date=         day-level summary (total calls, peak agents, etc.)
  GET /api/forecast?date=        demand forecast per 30-min slot
  GET /api/staffing?date=        staffing schedule per 30-min slot
  GET /api/weekly-forecast?week_start=  7-day forecast with error bars

How to run locally (no Docker):
  pip install fastapi uvicorn
  cd <project root>
  PYTHONPATH=src uvicorn main_module.api.main:app --reload --port 8000
  Then open: http://localhost:8000
  Interactive API docs: http://localhost:8000/docs
"""

import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pandas.tseries.holiday import USFederalHolidayCalendar

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = Path(os.environ.get(
    "MODEL_PATH",
    str(DATA_ROOT / "data" / "models" / "call_volume_model_bundle.pkl"),
))

# ---------------------------------------------------------------------------
# Major holidays — must match train_model.py exactly
# ---------------------------------------------------------------------------

MAJOR_HOLIDAYS = pd.to_datetime([
    "2024-01-01", "2025-01-01", "2026-01-01",
    "2024-11-28", "2025-11-27", "2026-11-26",
    "2024-12-25", "2025-12-25", "2026-12-25",
])

# ---------------------------------------------------------------------------
# Global variables
# ---------------------------------------------------------------------------

emulator = None
optimizer = None
model_ready = False

# ---------------------------------------------------------------------------
# Model loader — @lru_cache ensures the pkl is read exactly once
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_model_bundle():
    """
    Load the pre-trained RF+GB ensemble bundle from disk.

    Uses @lru_cache(maxsize=1) so the pickle file is deserialized exactly
    once: startup() pre-warms the cache, and all subsequent calls return
    the same Python objects instantly.

    Returns a dict with keys:
        rf_model, gb_model, features, holiday_profiles,
        forecast_weeks, daily_std_lookup, call_volume_history, trained_at
    """
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            f"Run 'PYTHONPATH=src python scripts/train_model.py' first."
        )
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    print(f"Model bundle loaded from {MODEL_PATH}")
    return bundle


def _build_emulator_and_optimizer():
    """Initialize the queueing emulator and staffing optimizer."""
    from main_module.workforce.call_center_emulator import CallCenterEmulator, EmulatorConfig
    from main_module.workforce.supply_optimizer import SupplyOptimizer

    emu = CallCenterEmulator(EmulatorConfig(
        avg_handle_time=600,             # 10-min average call handle time
        sla_threshold_seconds=60,        # SLA = answered within 60 seconds
        avg_patience_time=180,           # callers hang up after ~3 min wait
        interval_duration_seconds=1800,  # 30-min slot
    ))
    opt = SupplyOptimizer(emu, max_supply=5000)
    return emu, opt


# ---------------------------------------------------------------------------
# Create the FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Workforce Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup — runs once automatically when the server starts
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup():
    """
    Load the pre-trained model bundle and initialize the Erlang-A optimizer.

    No training happens here — the model is loaded from the pkl file
    produced by scripts/train_model.py.
    """
    global emulator, optimizer, model_ready

    print("=" * 60)
    print("Startup: loading pre-trained model bundle...")
    print("=" * 60)

    try:
        load_model_bundle()  # pre-warm the @lru_cache

        emulator, optimizer = _build_emulator_and_optimizer()

        model_ready = True
        print("=" * 60)
        print("Startup complete! Model loaded, API is ready.")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("API will return placeholder data.")
    except Exception as e:
        print(f"WARNING: Startup failed: {e}")
        print("API will return placeholder data.")


@app.on_event("shutdown")
def shutdown():
    print("Server shutting down.")


# ---------------------------------------------------------------------------
# Helper: placeholder data
# ---------------------------------------------------------------------------

def get_placeholder_slots():
    """Return zero-filled slots when the model bundle is not available."""
    time_slots = [f"{h:02d}:{m:02d}" for h in range(5, 17) for m in (0, 30)]
    return [
        {
            "time": t,
            "predicted_calls": 0,
            "agents": 0,
            "avg_wait_time": 0,
            "sla_compliance": 0,
            "utilization_rate": 0,
            "abandonment_rate": 0,
            "is_feasible": False,
        }
        for t in time_slots
    ]


# ---------------------------------------------------------------------------
# Helper: run the full ML pipeline for one date
# ---------------------------------------------------------------------------

def run_pipeline_for_date(date_str, min_sla, max_wait_time, max_occupancy):
    """
    Run the RF+GB ensemble + Erlang-A optimizer for a given date.

    For each 30-min slot (05:00–17:00):
      1. Build the 15-feature vector (calendar + lag features)
      2. Predict call volume with both RF and GB, average 50/50
      3. Apply holiday override if the date is a major holiday
      4. Feed predicted calls into SupplyOptimizer (binary search + Erlang-A)

    Returns a list of 24 dicts, one per slot.
    """
    from main_module.workforce.supply_optimizer import OptimizationConstraints

    bundle = load_model_bundle()
    rf_model = bundle["rf_model"]
    gb_model = bundle["gb_model"]
    features = bundle["features"]
    holiday_profiles = bundle["holiday_profiles"]
    forecast_weeks = bundle["forecast_weeks"]
    history = bundle["call_volume_history"]
    aht_lookup = bundle.get("aht_lookup", {})

    constraints = OptimizationConstraints(
        min_sla=min_sla,
        max_wait_time=max_wait_time,
        max_occupancy=max_occupancy,
    )

    ts = pd.Timestamp(date_str)
    dow = ts.dayofweek
    tax_day = pd.Timestamp(f"{ts.year}-04-15")
    days_to_tax = (tax_day - ts.normalize()).days

    # Pre-compute holiday flags for this date (once, not per slot)
    cal = USFederalHolidayCalendar()
    date_holidays = cal.holidays(start=ts.normalize(), end=ts.normalize())
    date_normalized = ts.normalize()
    is_major = int(date_normalized in MAJOR_HOLIDAYS)
    is_minor = int(len(date_holidays) > 0 and not is_major)

    # Lag feature helpers
    lag_intervals = forecast_weeks * 7 * 48
    history_mean = float(history.mean())
    history_max = float(history.max())

    interval = optimizer.emulator.config.interval_duration_seconds
    default_aht = optimizer.emulator.config.avg_handle_time

    results = []
    for hour in range(5, 17):
        for minute in (0, 30):
            slot_str = f"{hour:02d}:{minute:02d}"
            slot_ts = pd.Timestamp(f"{date_str} {slot_str}")

            # Compute lag features from historical call volume
            lag_ts = slot_ts - pd.Timedelta(weeks=forecast_weeks)
            lag_value = history.get(lag_ts, history_mean)

            window_start = lag_ts - pd.Timedelta(weeks=forecast_weeks)
            window = history.loc[window_start:lag_ts]
            trend_value = float(window.mean()) if len(window) > 0 else history_mean
            max_value = float(window.max()) if len(window) > 0 else history_max

            # Build the 15-feature vector matching training
            lag_feat = f"lag_{forecast_weeks}weeks"
            trend_feat = f"trend_{forecast_weeks}w"
            max_feat = f"max_{forecast_weeks}w"

            feat_row = pd.DataFrame([{
                "hour_sin":         np.sin(2 * np.pi * hour / 24),
                "hour_cos":         np.cos(2 * np.pi * hour / 24),
                "day_sin":          np.sin(2 * np.pi * dow / 7),
                "day_cos":          np.cos(2 * np.pi * dow / 7),
                "month":            ts.month,
                "weekofyear":       int(ts.isocalendar()[1]),
                "is_january":       int(ts.month == 1),
                "is_minor_holiday": is_minor,
                "is_major_holiday": is_major,
                "days_to_tax_day":  days_to_tax,
                "is_tax_season":    int(ts.month <= 4 and days_to_tax >= 0),
                "is_post_tax_drop": int(days_to_tax < 0 and days_to_tax > -31),
                lag_feat:           float(lag_value),
                trend_feat:         float(trend_value),
                max_feat:           float(max_value),
            }])

            # Ensemble prediction: 50/50 average of RF and GB
            pred_rf = rf_model.predict(feat_row[features])[0]
            pred_gb = gb_model.predict(feat_row[features])[0]
            predicted_calls = max(0, int(round((pred_rf + pred_gb) / 2)))

            # Business logic override: major holidays use historical profile
            if is_major:
                key = (slot_ts.month, slot_ts.day, slot_ts.time())
                predicted_calls = int(round(holiday_profiles.get(key, 15)))

            # Staffing optimization via Erlang-A
            slot_aht = aht_lookup.get((dow, slot_str), default_aht)
            traffic_erlangs = predicted_calls * slot_aht / interval
            min_agents_floor = max(1, int(np.ceil(traffic_erlangs / max_occupancy)) + 1)

            supply_result = optimizer.optimize(
                predicted_calls, constraints,
                min_agents=min_agents_floor,
                avg_handle_time=slot_aht,
            )
            metrics = supply_result.predicted_metrics

            results.append({
                "time":             slot_str,
                "predicted_calls":  predicted_calls,
                "agents":           supply_result.headcount,
                "avg_wait_time":    round(metrics.avg_wait_time, 1),
                "sla_compliance":   round(metrics.sla_compliance, 1),
                "utilization_rate": round(metrics.utilization_rate, 1),
                "abandonment_rate": round(metrics.abandonment_rate, 1),
                "is_feasible":      supply_result.is_feasible,
                "aht_seconds_used": float(round(slot_aht, 1)),
            })

    return results


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Health check — confirms server is running and model is loaded."""
    return {
        "status": "ok",
        "model_ready": model_ready,
        "model_path": str(MODEL_PATH),
        "message": "Workforce Optimization API is running",
    }


@app.get("/api/metrics")
def get_metrics(date: str = "2025-04-15"):
    """
    Return day-level summary metrics for the given date.
    Powers the 4 KPI cards at the top of the React dashboard.
    """
    if not model_ready:
        slots = get_placeholder_slots()
    else:
        try:
            slots = run_pipeline_for_date(
                date_str=date,
                min_sla=0.80,
                max_wait_time=60.0,
                max_occupancy=0.85,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if not slots:
        raise HTTPException(
            status_code=404,
            detail=f"No open business-hours intervals found for {date}",
        )

    total_calls = sum(s["predicted_calls"] for s in slots)
    peak_agents = max(s["agents"] for s in slots)
    avg_sla = round(sum(s["sla_compliance"] for s in slots) / len(slots), 1)
    avg_wait = round(sum(s["avg_wait_time"] for s in slots) / len(slots), 1)
    avg_occupancy = round(sum(s["utilization_rate"] for s in slots) / len(slots), 1)
    feasible_count = sum(1 for s in slots if s["is_feasible"])

    return {
        "date": date,
        "total_calls": total_calls,
        "peak_agents": peak_agents,
        "avg_sla_compliance": avg_sla,
        "avg_wait_time": avg_wait,
        "avg_occupancy": avg_occupancy,
        "feasible_intervals": feasible_count,
        "total_intervals": len(slots),
        "model_ready": model_ready,
    }


@app.get("/api/forecast")
def get_forecast(date: str = "2025-04-15"):
    """
    Return predicted call volume per 30-min slot for the given date.
    Powers the DemandChart (area chart) in the React dashboard.
    """
    if not model_ready:
        slots = get_placeholder_slots()
        return [{"time": s["time"], "predicted_calls": s["predicted_calls"], "model_used": "placeholder"} for s in slots]

    try:
        slots = run_pipeline_for_date(date, 0.80, 60.0, 0.85)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return [{"time": s["time"], "predicted_calls": s["predicted_calls"], "model_used": "rf_gb_ensemble"} for s in slots]


@app.get("/api/weekly-forecast")
def get_weekly_forecast(week_start: str = "2025-04-14"):
    """
    Return 7 days of predicted total call volume for the given week.
    Powers the Weekly Demand Forecast bar chart in the React dashboard.
    """
    if not model_ready:
        return []

    try:
        bundle = load_model_bundle()
        daily_std_lookup = bundle["daily_std_lookup"]

        start = pd.Timestamp(week_start)
        result = []
        for i in range(7):
            day = start + pd.Timedelta(days=i)
            date_str = day.strftime("%Y-%m-%d")
            slots = run_pipeline_for_date(date_str, 0.80, 60.0, 0.85)
            total_calls = sum(s["predicted_calls"] for s in slots)
            dow = day.dayofweek
            std = daily_std_lookup.get(dow, 1000)
            result.append({
                "date":        date_str,
                "day_label":   f"{day.strftime('%a')} {day.month}/{day.day}",
                "total_calls": total_calls,
                "range":       round(std),
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/staffing")
def get_staffing(
    date: str = "2025-04-15",
    min_sla: float = 0.80,
    max_wait: float = 60.0,
    max_occupancy: float = 0.85,
):
    """
    Return the full staffing schedule per 30-min slot for the given date.
    Powers the staffing table and combined chart in the React dashboard.
    """
    if not model_ready:
        return get_placeholder_slots()

    try:
        return run_pipeline_for_date(
            date_str=date,
            min_sla=min_sla,
            max_wait_time=max_wait,
            max_occupancy=max_occupancy,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
