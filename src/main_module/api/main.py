"""
Workforce Optimization API
--------------------------
FastAPI backend that serves ML pipeline results to the React dashboard.

Endpoints:
  GET /                          health check - confirms server is running
  GET /api/metrics?date=         day-level summary (total calls, peak agents, etc.)
  GET /api/forecast?date=        demand forecast per 30-min slot
  GET /api/staffing?date=        staffing schedule per 30-min slot

How to run locally (no Docker):
  pip install fastapi uvicorn
  cd <project root>
  PYTHONPATH=src uvicorn src.main_module.api.main:app --reload --port 8000
  Then open: http://localhost:8000
  Interactive API docs: http://localhost:8000/docs
"""

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

DATA_ROOT      = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH      = DATA_ROOT / "data" / "interim" / "mock_intuit_2year_data.csv"
DATASET_1_PATH = DATA_ROOT / "data" / "parquet" / "dataset_1_call_related.parquet"
DATASET_4_PATH = DATA_ROOT / "data" / "parquet" / "dataset_4_expert_state_interval.parquet"

# ---------------------------------------------------------------------------
# Global variables
# Hold the trained model so we only train ONCE at startup,
# not on every single request.
# ---------------------------------------------------------------------------

# These start as None — they get filled in by startup() below
forecaster = None
emulator = None
optimizer = None
model_ready = False  # becomes True once training finishes successfully

# MVP: XGBoost model + historical lookup tables from parquet files
xgb_model        = None  # XGBoost trained on dataset_1 call volume
xgb_features     = []    # feature column names used during training
wait_time_lookup = {}    # (day_of_week, slot) -> avg wait time (seconds)
sla_lookup       = {}    # (day_of_week, slot) -> avg SLA %
occupancy_lookup = {}    # (day_of_week, slot) -> avg occupancy %
agents_lookup    = {}    # (day_of_week, slot) -> avg agent count
daily_std_lookup = {}    # day_of_week -> historical std dev of daily call volume (for error bars)


# ---------------------------------------------------------------------------
# Create the FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Workforce Optimization API")

# CORS middleware — allows the React frontend (localhost:5173) to call this API.
# Without this, the browser blocks all requests from a different port (CORS error).
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
    Train the ML model when the server starts up.

    This function runs automatically when you start the server.
    Training takes about 1 minute on first run.
    After that, the trained model stays in memory so every API request is fast.

    The `global` keyword lets us write to the module-level variables
    (forecaster, emulator, optimizer, model_ready) defined above.
    Without `global`, Python would treat them as local variables and
    the rest of the code would not be able to see the trained model.
    """
    global forecaster, emulator, optimizer, model_ready
    global xgb_model, xgb_features, wait_time_lookup, sla_lookup, occupancy_lookup, agents_lookup, daily_std_lookup

    # --- MVP: XGBoost + parquet lookup tables ---
    # Trains XGBoost on dataset_1 call volume; builds lookup tables from dataset_1 and dataset_4.
    # Mirrors feature engineering from scripts/forecasting_dynamic_weeks.py (no lag features).
    print("=" * 60)
    print("Startup: loading parquet data and training XGBoost...")
    print("(This takes ~30 seconds — only happens once at startup)")
    print("=" * 60)

    try:
        from xgboost import XGBRegressor

        # --- dataset_1: train XGBoost + compute wait time / SLA lookups ---
        # Only load the two timestamp columns we need — avoids reading the full 557 MB file
        call_df = pd.read_parquet(str(DATASET_1_PATH), columns=["arrival_time_utc", "start_time_utc"])
        call_df["arrival"] = pd.to_datetime(call_df["arrival_time_utc"])
        call_df["start"]   = pd.to_datetime(call_df["start_time_utc"])

        # Aggregate to 30-min call volume (mirrors forecasting_dynamic_weeks.py step 1)
        df_agg = call_df.set_index("arrival").resample("30min").size().to_frame("call_volume")
        df_agg = df_agg.asfreq("30min", fill_value=0)

        # Feature engineering — same columns as forecasting_dynamic_weeks.py, no lag features
        df_agg["hour"]      = df_agg.index.hour
        df_agg["dayofweek"] = df_agg.index.dayofweek
        df_agg["month"]     = df_agg.index.month
        df_agg["hour_sin"]  = np.sin(2 * np.pi * df_agg["hour"] / 24)
        df_agg["hour_cos"]  = np.cos(2 * np.pi * df_agg["hour"] / 24)
        df_agg["day_sin"]   = np.sin(2 * np.pi * df_agg["dayofweek"] / 7)
        df_agg["day_cos"]   = np.cos(2 * np.pi * df_agg["dayofweek"] / 7)
        tax_days            = pd.to_datetime(df_agg.index.year.astype(str) + "-04-15")
        df_agg["days_to_tax"]   = (tax_days - df_agg.index.normalize()).days
        df_agg["is_tax_season"] = ((df_agg["month"] <= 4) & (df_agg["days_to_tax"] >= 0)).astype(int)

        xgb_features = ["hour_sin", "hour_cos", "day_sin", "day_cos",
                         "month", "days_to_tax", "is_tax_season"]

        train = df_agg.dropna()
        xgb_model = XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            n_jobs=-1, random_state=42
        )
        xgb_model.fit(train[xgb_features], train["call_volume"])
        print("XGBoost trained on call volume.")

        # Daily std dev lookup — used for error bars in the weekly forecast chart
        df_agg["date_only"] = df_agg.index.normalize()
        df_agg["dow_day"]   = df_agg.index.dayofweek
        daily_totals     = df_agg.groupby(["dow_day", "date_only"])["call_volume"].sum().reset_index()
        daily_std_lookup = daily_totals.groupby("dow_day")["call_volume"].std().to_dict()
        print("Daily std dev lookup built for weekly chart error bars.")

        # --- Initialize SupplyOptimizer (Erlang-A) ---
        from main_module.workforce.call_center_emulator import CallCenterEmulator, EmulatorConfig
        from main_module.workforce.supply_optimizer import SupplyOptimizer
        emulator = CallCenterEmulator(EmulatorConfig(
            avg_handle_time=600,             # 10-min average call handle time
            sla_threshold_seconds=60,        # SLA = answered within 60 seconds
            avg_patience_time=180,           # callers hang up after ~3 min wait
            interval_duration_seconds=1800,  # 30-min slot
        ))
        optimizer = SupplyOptimizer(emulator, max_supply=5000)
        print("SupplyOptimizer (Erlang-A) initialized.")

        model_ready = True
        print("=" * 60)
        print("Startup complete! API is ready to serve real ML predictions.")
        print("=" * 60)

    except Exception as e:
        print(f"WARNING: Startup training failed: {e}")
        print("API will return placeholder data.")

    # --- OLD ML TRAINING (HybridForecaster) — kept for reference ---
    # print("Startup: skipping model training, using placeholder data.")
    # print("To enable real ML results, uncomment the training block in startup().")

    # if not DATA_PATH.exists():
    #     print(f"WARNING: Data file not found at {DATA_PATH}")
    #     print("The API will return placeholder data until the file is available.")
    #     return

    # # Import the ML modules from our Python package
    # from main_module.workforce import (
    #     CallCenterEmulator,
    #     EmulatorConfig,
    #     HybridForecaster,
    #     SupplyOptimizer,
    # )

    # # Step 1: Train the demand forecasting model
    # print("=" * 60)
    # print("Startup: training HybridForecaster...")
    # print("(This takes ~1 minute — only happens once at startup)")
    # print("=" * 60)

    # forecaster = HybridForecaster()
    # forecaster.train(
    #     str(DATA_PATH),
    #     test_year=2025,
    #     tune_hyperparameters=True,
    #     n_trials=10,
    # )

    # # Step 2: Set up the call center emulator with business config
    # emulator = CallCenterEmulator(
    #     config=EmulatorConfig(
    #         avg_handle_time=300,             # average call handle time = 5 minutes
    #         sla_threshold_seconds=60,        # SLA = must answer within 60 seconds
    #         interval_duration_seconds=1800,  # each time slot = 30 minutes
    #     )
    # )

    # # Step 3: Set up the supply optimizer (uses the emulator internally)
    # optimizer = SupplyOptimizer(emulator, max_supply=500)

    # model_ready = True
    # print("=" * 60)
    # print("Startup complete! API is ready.")
    # print("=" * 60)


# ---------------------------------------------------------------------------
# Shutdown — runs when the server stops
# ---------------------------------------------------------------------------

@app.on_event("shutdown")
def shutdown():
    print("Server shutting down.")


# ---------------------------------------------------------------------------
# Helper: placeholder data
# Used when the data CSV file is missing (e.g. fresh clone without data folder)
# ---------------------------------------------------------------------------

def get_placeholder_slots():
    """
    Return fake per-slot data so the server does not crash when the real
    data file is not present on this machine.

    A teammate can still run the server and see something in the React
    dashboard even before they have the data file downloaded.

    Once the real data file exists, this function is never called.
    """
    import random
    random.seed(42)  # seed makes the random numbers the same every time

    # Business hours: 5am to 5pm in 30-minute slots  →  24 slots total
    time_slots = [f"{h:02d}:{m:02d}" for h in range(5, 17) for m in (0, 30)]

    slots = []
    for t in time_slots:
        # NOTE: all zeros = model not ready / parquet files not found.
        # If you see zeros in the dashboard, it means startup() failed or is still running.
        slots.append({
            "time": t,
            "predicted_calls": 0,
            "agents": 0,
            "avg_wait_time": 0,
            "sla_compliance": 0,
            "utilization_rate": 0,
            "abandonment_rate": 0,
            "is_feasible": False,
        })
    return slots


# ---------------------------------------------------------------------------
# Helper: run the full ML pipeline for one date
# ---------------------------------------------------------------------------

def run_pipeline_for_date(date_str, min_sla, max_wait_time, max_occupancy):
    """
    Run the full ML pipeline for a given date and constraints.

    This is the core logic that connects the forecaster and optimizer.
    It is called by the /api/metrics and /api/staffing endpoints.

    How it works:
      1. Pass date_str to forecaster.predict_day()
         The forecaster uses the seasonal/time patterns it learned during
         training to estimate how many calls will arrive each 30-min slot
         on that specific date. Works for any date: past, present, or future.

      2. For each open slot, pass the predicted call count to optimizer.optimize()
         The optimizer tries 1 agent, then 2, then 3... until the SLA/wait/
         occupancy constraints are met, then returns the minimum headcount needed.

      3. Return all slots as a list of dicts (one dict per 30-min slot)

    Parameters:
      date_str     : date string like "2025-04-15"
                     comes from the ?date= URL parameter sent by React
      min_sla      : minimum SLA as a decimal fraction  (e.g. 0.80 = 80%)
                     comes from the SLA slider in the React SimulationPanel
      max_wait_time: max average wait time in seconds   (e.g. 60.0)
                     comes from the Max Wait slider in React
      max_occupancy: max agent occupancy as a fraction  (e.g. 0.85 = 85%)
                     comes from the Max Occupancy slider in React
    """
    from main_module.workforce.supply_optimizer import OptimizationConstraints

    constraints = OptimizationConstraints(
        min_sla=min_sla,
        max_wait_time=max_wait_time,
        max_occupancy=max_occupancy,
    )

    ts  = pd.Timestamp(date_str)
    dow = ts.dayofweek
    tax_day     = pd.Timestamp(f"{ts.year}-04-15")
    days_to_tax = (tax_day - ts.normalize()).days

    aht      = optimizer.emulator.config.avg_handle_time
    interval = optimizer.emulator.config.interval_duration_seconds

    results = []
    for hour in range(5, 17):
        for minute in (0, 30):
            # Build one-row feature vector for XGBoost (same features as training)
            feat_row = pd.DataFrame([{
                "hour_sin":      np.sin(2 * np.pi * hour / 24),
                "hour_cos":      np.cos(2 * np.pi * hour / 24),
                "day_sin":       np.sin(2 * np.pi * dow / 7),
                "day_cos":       np.cos(2 * np.pi * dow / 7),
                "month":         ts.month,
                "days_to_tax":   days_to_tax,
                "is_tax_season": int(ts.month <= 4 and days_to_tax >= 0),
            }])
            predicted_calls = max(0, int(round(xgb_model.predict(feat_row[xgb_features])[0])))

            # Skip low-agent counts that can never satisfy the occupancy constraint.
            # traffic_erlangs = offered load; need agents > traffic / max_occupancy.
            traffic_erlangs  = predicted_calls * aht / interval
            min_agents_floor = max(1, int(np.ceil(traffic_erlangs / max_occupancy)) + 1)

            supply_result = optimizer.optimize(predicted_calls, constraints, min_agents=min_agents_floor)
            metrics = supply_result.predicted_metrics

            results.append({
                "time":             f"{hour:02d}:{minute:02d}",
                "predicted_calls":  predicted_calls,
                "agents":           supply_result.headcount,
                "avg_wait_time":    round(metrics.avg_wait_time, 1),
                "sla_compliance":   round(metrics.sla_compliance, 1),
                "utilization_rate": round(metrics.utilization_rate, 1),
                "abandonment_rate": round(metrics.abandonment_rate, 1),
                "is_feasible":      supply_result.is_feasible,
            })

    return results

    # --- OLD pipeline using HybridForecaster + SupplyOptimizer (kept for reference) ---
    # from main_module.workforce import OptimizationConstraints
    # constraints = OptimizationConstraints(
    #     min_sla=min_sla,
    #     max_wait_time=max_wait_time,
    #     max_occupancy=max_occupancy,
    # )
    # demand_df = forecaster.predict_day(date_str)
    # demand_df["time"] = demand_df["interval_start"].dt.strftime("%H:%M")
    # demand_df["is_open"] = (
    #     (demand_df["interval_start"].dt.hour >= 5)
    #     & (demand_df["interval_start"].dt.hour < 17)
    # )
    # open_slots = demand_df[demand_df["is_open"] & (demand_df["predicted_calls"] > 0)]
    # results = []
    # for _, row in open_slots.iterrows():
    #     demand = int(row["predicted_calls"])
    #     supply_result = optimizer.optimize(demand, constraints)
    #     metrics = supply_result.predicted_metrics
    #     results.append({
    #         "time": row["time"],
    #         "predicted_calls": demand,
    #         "agents": supply_result.headcount,
    #         "avg_wait_time": round(metrics.avg_wait_time, 1),
    #         "sla_compliance": round(metrics.sla_compliance, 1),
    #         "utilization_rate": round(metrics.utilization_rate, 1),
    #         "abandonment_rate": round(metrics.abandonment_rate, 1),
    #         "is_feasible": supply_result.is_feasible,
    #     })
    # return results


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """
    Health check endpoint.
    Visit http://localhost:8000 in your browser to confirm the server is running.
    Returns whether the ML model has finished training yet.
    """
    return {
        "status": "ok",
        "model_ready": model_ready,
        "message": "Workforce Optimization API is running",
    }


@app.get("/api/metrics")
def get_metrics(date: str = "2025-04-15"):
    """
    Return day-level summary metrics for the given date.

    These numbers power the 4 KPI cards at the top of the React dashboard:
      total_calls        →  Total Calls Processed card
      avg_sla_compliance →  Service Level (SLA) card
      avg_wait_time      →  Avg. Waiting Time card
      peak_agents        →  peak headcount needed that day

    How React calls this:
      fetch("http://localhost:8000/api/metrics?date=2025-04-15")

    The date comes from whatever the user selected in the React date picker.
    The default "2025-04-15" is only used on the very first load before the
    user has picked a date.
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

    # Aggregate all per-slot numbers into a single day-level summary
    total_calls   = sum(s["predicted_calls"] for s in slots)
    peak_agents   = max(s["agents"] for s in slots)
    avg_sla       = round(sum(s["sla_compliance"] for s in slots) / len(slots), 1)
    avg_wait      = round(sum(s["avg_wait_time"] for s in slots) / len(slots), 1)
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

    This powers the DemandChart (area chart) in the React dashboard.

    How React calls this:
      fetch("http://localhost:8000/api/forecast?date=2025-04-15")

    The forecaster applies the seasonal patterns it learned during training
    to estimate call volume for that specific date. Works for any date.

    Example response:
      [
        {"time": "05:00", "predicted_calls": 12, "model_used": "long-term"},
        {"time": "05:30", "predicted_calls": 18, "model_used": "long-term"},
        ...
      ]
    """
    if not model_ready:
        slots = get_placeholder_slots()
        return [{"time": s["time"], "predicted_calls": s["predicted_calls"], "model_used": "placeholder"} for s in slots]

    try:
        slots = run_pipeline_for_date(date, 0.80, 60.0, 0.85)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return [{"time": s["time"], "predicted_calls": s["predicted_calls"], "model_used": "xgboost"} for s in slots]

    # --- OLD forecast using HybridForecaster (kept for reference) ---
    # demand_df = forecaster.predict_day(date)
    # demand_df["time"] = demand_df["interval_start"].dt.strftime("%H:%M")
    # demand_df["is_open"] = (
    #     (demand_df["interval_start"].dt.hour >= 5)
    #     & (demand_df["interval_start"].dt.hour < 17)
    # )
    # open_df = demand_df[demand_df["is_open"]]
    # return [
    #     {"time": row["time"], "predicted_calls": int(row["predicted_calls"]), "model_used": row["model_used"]}
    #     for _, row in open_df.iterrows()
    # ]


@app.get("/api/weekly-forecast")
def get_weekly_forecast(week_start: str = "2025-04-14"):
    """
    Return 7 days of XGBoost-predicted total call volume for the given week.
    Powers the Weekly Demand Forecast bar chart in the React dashboard.

    How React calls this:
      fetch("http://localhost:8000/api/weekly-forecast?week_start=2025-04-14")

    week_start should be a Monday (YYYY-MM-DD). Returns 7 items, one per day Mon–Sun.
    `range` is the historical std dev of daily call volume for that day-of-week,
    used as the ± error bar in the bar chart.

    Example response:
      [
        {"date": "2025-04-14", "day_label": "Mon 4/14", "total_calls": 27500, "range": 1800},
        {"date": "2025-04-15", "day_label": "Tue 4/15", "total_calls": 28100, "range": 1800},
        ...
      ]
    """
    if not model_ready:
        return []

    try:
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

    This powers the staffing table and combined chart in the React dashboard.
    The SimulationPanel sliders in React pass their values as query parameters
    so users can re-run the optimizer with different constraints interactively.

    How React calls this (default constraints):
      fetch("http://localhost:8000/api/staffing?date=2025-04-15")

    How React calls this (custom constraints from sliders):
      fetch("http://localhost:8000/api/staffing?date=2025-04-15&min_sla=0.90&max_wait=45&max_occupancy=0.80")

    Query parameters — all come from the React UI:
      date         : date string from the date picker,   e.g. "2025-04-15"
      min_sla      : SLA slider value as fraction,        e.g. 0.80 = 80%
      max_wait     : Max Wait slider value in seconds,    e.g. 60.0
      max_occupancy: Max Occupancy slider as fraction,    e.g. 0.85 = 85%

    Example response:
      [
        {
          "time": "09:00",
          "predicted_calls": 87,
          "agents": 42,
          "avg_wait_time": 18.3,
          "sla_compliance": 91.5,
          "utilization_rate": 74.2,
          "abandonment_rate": 2.1,
          "is_feasible": true
        },
        ...
      ]
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
