"""
Workforce Optimization API
--------------------------
FastAPI backend that serves ML pipeline results to the React dashboard.

The CombinedForecaster model is trained offline by `scripts/train_model.py` and
saved to `data/models/call_volume_model_bundle.pkl`.  At startup this API
loads the bundle once via @lru_cache — no training happens here.

Endpoints:
  GET /                       health check
  GET /api/metrics?date=      day-level summary (total calls, peak agents)
  GET /api/forecast?date=     demand forecast per 30-min slot
  GET /api/staffing?date=     staffing schedule per 30-min slot
  GET /api/weekly-forecast    7-day forecast with error bars
  GET /api/agents             list agents with performance metrics
  GET /api/agents/{id}        detailed profile for a single agent
  GET /api/agents/segments    segment-level summary stats
  GET /api/schedule?date=     shift schedule for a date
  GET /api/routing/recommend  agent routing recommendations

How to run locally (no Docker):
  pip install fastapi uvicorn
  cd <project root>
  PYTHONPATH=src uvicorn main_module.api.main:app --reload --port 8000
  Then open: http://localhost:8000
  Interactive API docs: http://localhost:8000/docs
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from main_module.workforce.agent_analytics import AgentAnalytics
from main_module.workforce.agent_routing import AgentRouter
from main_module.workforce.combined_forecaster import CombinedForecaster
from main_module.workforce.shift_scheduler import ShiftScheduler

DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        str(DATA_ROOT / "data" / "models" / "call_volume_model_bundle.pkl"),
    )
)
PARQUET_DIR = DATA_ROOT / "data" / "parquet"

emulator = None
optimizer = None
model_ready = False
forecaster: Optional[CombinedForecaster] = None
agent_analytics: Optional[AgentAnalytics] = None
agent_router: Optional[AgentRouter] = None
shift_scheduler: Optional[ShiftScheduler] = None
_schedule_cache: dict = {}


@lru_cache(maxsize=1)
def load_model_bundle():
    """
    Load the pre-trained CombinedForecaster bundle from disk.

    Uses @lru_cache(maxsize=1) so the pickle file is deserialized exactly
    once: startup() pre-warms the cache, and all subsequent calls return
    the same Python objects instantly.
    """
    global forecaster
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            f"Run 'PYTHONPATH=src python scripts/train_model.py' first."
        )
    forecaster = CombinedForecaster()
    forecaster.load_model(str(MODEL_PATH))
    print(f"Model bundle loaded from {MODEL_PATH}")
    return forecaster


def _build_emulator_and_optimizer():
    from main_module.workforce.call_center_emulator import (
        CallCenterEmulator,
        EmulatorConfig,
    )
    from main_module.workforce.supply_optimizer import SupplyOptimizer

    emu = CallCenterEmulator(
        EmulatorConfig(
            avg_handle_time=600,
            sla_threshold_seconds=60,
            avg_patience_time=180,
            interval_duration_seconds=1800,
        )
    )
    opt = SupplyOptimizer(emu, max_supply=5000)
    return emu, opt


app = FastAPI(title="Workforce Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_agent_analytics():
    global agent_analytics
    data_dir = _resolve_data_dir()
    agent_analytics = AgentAnalytics(data_dir=data_dir)
    agent_analytics.load(tax_year=None)
    print(f"AgentAnalytics loaded from {data_dir}")
    return agent_analytics


def _resolve_data_dir():
    for d in [PARQUET_DIR, DATA_ROOT / "data" / "raw"]:
        p = Path(d)
        if (p / "dataset_4_expert_state_interval.parquet").exists():
            return str(p)
    return str(PARQUET_DIR)


@lru_cache(maxsize=1)
def load_agent_router():
    global agent_router
    data_dir = _resolve_data_dir()
    agent_router = AgentRouter(data_dir=data_dir)
    agent_router.load(tax_year=None)
    print(f"AgentRouter loaded from {data_dir}")
    return agent_router


@lru_cache(maxsize=1)
def load_shift_scheduler():
    global shift_scheduler
    data_dir = _resolve_data_dir()
    shift_scheduler = ShiftScheduler(data_dir=data_dir)
    shift_scheduler.load_agent_patterns(tax_year=None, recent_days=90)
    print(f"ShiftScheduler loaded from {data_dir}")
    return shift_scheduler


@app.on_event("startup")
def startup():
    global emulator, optimizer, model_ready

    print("=" * 60)
    print("Startup: loading pre-trained model bundle...")
    print("=" * 60)

    try:
        load_model_bundle()

        emulator, optimizer = _build_emulator_and_optimizer()

        load_agent_analytics()

        load_shift_scheduler()

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


def get_placeholder_slots():
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


@lru_cache(maxsize=32)
def _run_pipeline_for_date_cached(
    date_str, min_sla, max_wait_time, max_occupancy
):
    from main_module.workforce.supply_optimizer import OptimizationConstraints

    fc = load_model_bundle()
    aht_lookup = getattr(fc, "aht_lookup", {})

    constraints = OptimizationConstraints(
        min_sla=min_sla,
        max_wait_time=max_wait_time,
        max_occupancy=max_occupancy,
    )

    ts = pd.Timestamp(date_str)
    dow = ts.dayofweek

    interval = optimizer.emulator.config.interval_duration_seconds
    default_aht = optimizer.emulator.config.avg_handle_time

    results = []
    for hour in range(5, 17):
        for minute in (0, 30):
            slot_str = f"{hour:02d}:{minute:02d}"
            h_utc = (hour + 8) % 24
            slot_ts = pd.Timestamp(f"{date_str} {h_utc:02d}:{minute:02d}")

            predicted_calls = fc.predict(slot_ts)

            slot_aht = aht_lookup.get((dow, slot_str), default_aht)
            traffic_erlangs = predicted_calls * slot_aht / interval
            min_agents_floor = max(
                1, int(np.ceil(traffic_erlangs / max_occupancy)) + 1
            )

            supply_result = optimizer.optimize(
                predicted_calls,
                constraints,
                min_agents=min_agents_floor,
                avg_handle_time=slot_aht,
            )
            metrics = supply_result.predicted_metrics

            results.append(
                {
                    "time": slot_str,
                    "predicted_calls": predicted_calls,
                    "agents": supply_result.headcount,
                    "avg_wait_time": round(metrics.avg_wait_time, 1),
                    "sla_compliance": round(metrics.sla_compliance, 1),
                    "utilization_rate": round(metrics.utilization_rate, 1),
                    "abandonment_rate": round(metrics.abandonment_rate, 1),
                    "is_feasible": supply_result.is_feasible,
                    "aht_seconds_used": float(round(slot_aht, 1)),
                }
            )

    return results


def run_pipeline_for_date(date_str, min_sla, max_wait_time, max_occupancy):
    return list(
        _run_pipeline_for_date_cached(
            str(date_str),
            float(min_sla),
            float(max_wait_time),
            float(max_occupancy),
        )
    )


@app.get("/")
def root():
    return {
        "status": "ok",
        "model_ready": model_ready,
        "model_path": str(MODEL_PATH),
        "message": "Workforce Optimization API is running",
    }


@app.get("/api/metrics")
def get_metrics(date: str = "2025-04-15"):
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
            print(
                f"Metrics pipeline failed for {date}: {e}, using placeholder"
            )
            slots = get_placeholder_slots()

    if not slots:
        raise HTTPException(
            status_code=404,
            detail=f"No open business-hours intervals found for {date}",
        )

    total_calls = sum(s["predicted_calls"] for s in slots)
    peak_agents = max(s["agents"] for s in slots)
    avg_sla = round(sum(s["sla_compliance"] for s in slots) / len(slots), 1)
    avg_wait = round(sum(s["avg_wait_time"] for s in slots) / len(slots), 1)
    avg_occupancy = round(
        sum(s["utilization_rate"] for s in slots) / len(slots), 1
    )
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
    if not model_ready:
        slots = get_placeholder_slots()
        return [
            {
                "time": s["time"],
                "predicted_calls": s["predicted_calls"],
                "model_used": "placeholder",
            }
            for s in slots
        ]

    try:
        slots = run_pipeline_for_date(date, 0.80, 60.0, 0.85)
    except Exception as e:
        print(f"Forecast pipeline failed for {date}: {e}, using placeholder")
        slots = get_placeholder_slots()

    return [
        {
            "time": s["time"],
            "predicted_calls": s["predicted_calls"],
            "model_used": "combined_forecaster",
        }
        for s in slots
    ]


@app.get("/api/weekly-forecast")
def get_weekly_forecast(week_start: str = "2025-04-14"):
    if not model_ready:
        return []

    try:
        fc = load_model_bundle()
        daily_std_lookup = getattr(fc, "daily_std_lookup", {})

        start = pd.Timestamp(week_start)
        result = []
        for i in range(7):
            day = start + pd.Timedelta(days=i)
            date_str = day.strftime("%Y-%m-%d")
            slots = run_pipeline_for_date(date_str, 0.80, 60.0, 0.85)
            total_calls = sum(s["predicted_calls"] for s in slots)
            dow = day.dayofweek
            std = daily_std_lookup.get(dow, 1000)
            result.append(
                {
                    "date": date_str,
                    "day_label": f"{day.strftime('%a')} {day.month}/{day.day}",
                    "total_calls": total_calls,
                    "range": round(std),
                }
            )
        return result
    except Exception as e:
        msg = f"Weekly forecast failed for {week_start}: {e}, using empty"
        print(msg)
        return []


@app.get("/api/staffing")
def get_staffing(
    date: str = "2025-04-15",
    min_sla: float = 0.80,
    max_wait: float = 60.0,
    max_occupancy: float = 0.85,
):
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
        print(f"Staffing pipeline failed for {date}: {e}, using placeholder")
        return get_placeholder_slots()


def safe_float(value, default=0.0):
    if pd.isna(value) or value is None:
        return default
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    if pd.isna(value) or value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


@app.get("/api/agents")
def get_agents(
    n: int = Query(100, description="Number of agents to return"),
    segment: Optional[str] = Query(
        None, description="Filter by expert segment"
    ),
    sort_by: str = Query("resolution_rate", description="Sort by field"),
    ascending: bool = Query(False, description="Sort ascending"),
):
    try:
        aa = load_agent_analytics()
        df = aa.top_performers(
            n=n, segment=segment, sort_by=sort_by, ascending=ascending
        )

        def format_aht(seconds):
            if pd.isna(seconds) or seconds is None:
                return "N/A"
            try:
                s = float(seconds)
                if np.isnan(s) or np.isinf(s):
                    return "N/A"
                m = int(s // 60)
                sec = int(s % 60)
                return f"{m}m {sec:02d}s"
            except (ValueError, TypeError):
                return "N/A"

        agents = []
        for _, row in df.iterrows():
            agents.append(
                {
                    "expert_id": str(row["expert_id"]),
                    "name": f"Agent {str(row['expert_id'])[-4:]}",
                    "segment": str(
                        row.get("expert_segment", "Unknown") or "Unknown"
                    ),
                    "business_segment": str(
                        row.get("business_segment", "Unknown") or "Unknown"
                    ),
                    "status": "Online",
                    "contacts": safe_int(row.get("contacts", 0)),
                    "answered_contacts": safe_int(
                        row.get("answered_contacts", 0)
                    ),
                    "resolution_rate": round(
                        safe_float(row.get("resolution_rate", 0)), 1
                    ),
                    "transfer_rate": round(
                        safe_float(row.get("transfer_rate", 0)), 1
                    ),
                    "aht": format_aht(
                        row.get("average_handle_time_seconds", 0)
                    ),
                    "aht_seconds": safe_float(
                        row.get("average_handle_time_seconds", 0)
                    ),
                    "hold_time_seconds": safe_float(
                        row.get("average_hold_time_seconds", 0)
                    ),
                    "composite_score": round(
                        safe_float(row.get("composite_score", 0)), 1
                    ),
                    "fcr_rate": round(
                        safe_float(
                            row.get("fcr_rate", row.get("resolution_rate", 0))
                        ),
                        1,
                    ),
                    "utilization": round(
                        safe_float(
                            row.get(
                                "utilization", row.get("mean_occupancy", 50)
                            ),
                            50,
                        ),
                        1,
                    ),
                    "mean_occupancy": round(
                        safe_float(row.get("mean_occupancy", 50), 50), 1
                    ),
                }
            )
        return agents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents/segments")
def get_agent_segments():
    try:
        aa = load_agent_analytics()
        summary = aa.segment_summary().reset_index()

        segments = []
        for _, row in summary.iterrows():
            segments.append(
                {
                    "segment": str(row["expert_segment"]),
                    "agent_count": safe_int(row["agent_count"]),
                    "mean_resolution": round(
                        safe_float(row["mean_resolution"]), 1
                    ),
                    "mean_transfer": round(
                        safe_float(row["mean_transfer"]), 1
                    ),
                    "mean_handle_time": round(
                        safe_float(row["mean_handle_time"]), 1
                    ),
                    "mean_contacts": round(
                        safe_float(row["mean_contacts"]), 1
                    ),
                    "mean_composite": round(
                        safe_float(row["mean_composite"]), 1
                    ),
                }
            )
        return segments
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents/{expert_id}")
def get_agent_profile(expert_id: str):
    try:
        aa = load_agent_analytics()
        profile = aa.agent_profile(expert_id)

        if profile is None:
            raise HTTPException(
                status_code=404, detail=f"Agent {expert_id} not found"
            )

        def format_aht(seconds):
            if pd.isna(seconds) or seconds is None:
                return "N/A"
            try:
                s = float(seconds)
                if np.isnan(s) or np.isinf(s):
                    return "N/A"
                m = int(s // 60)
                sec = int(s % 60)
                return f"{m}m {sec:02d}s"
            except (ValueError, TypeError):
                return "N/A"

        return {
            "expert_id": str(profile.get("expert_id", expert_id)),
            "name": f"Agent {str(profile.get('expert_id', expert_id))[-4:]}",
            "segment": str(
                profile.get("expert_segment", "Unknown") or "Unknown"
            ),
            "business_segment": str(
                profile.get("business_segment", "Unknown") or "Unknown"
            ),
            "status": "Online",
            "contacts": safe_int(profile.get("contacts", 0)),
            "answered_contacts": safe_int(profile.get("answered_contacts", 0)),
            "resolution_rate": round(
                safe_float(profile.get("resolution_rate", 0)), 1
            ),
            "transfer_rate": round(
                safe_float(profile.get("transfer_rate", 0)), 1
            ),
            "aht": format_aht(profile.get("average_handle_time_seconds", 0)),
            "aht_seconds": safe_float(
                profile.get("average_handle_time_seconds", 0)
            ),
            "hold_time_seconds": safe_float(
                profile.get("average_hold_time_seconds", 0)
            ),
            "composite_score": round(
                safe_float(profile.get("composite_score", 0)), 1
            ),
            "fcr_rate": round(
                safe_float(
                    profile.get("fcr_rate", profile.get("resolution_rate", 0))
                ),
                1,
            ),
            "utilization": round(
                safe_float(
                    profile.get(
                        "utilization", profile.get("mean_occupancy", 50)
                    ),
                    50,
                ),
                1,
            ),
            "mean_occupancy": round(
                safe_float(profile.get("mean_occupancy", 50), 50), 1
            ),
            "answer_rate": round(safe_float(profile.get("answer_rate", 0)), 1),
            "median_hold": round(safe_float(profile.get("median_hold", 0)), 1),
            "mean_hold": round(safe_float(profile.get("mean_hold", 0)), 1),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _df_to_assignment_dicts(schedule_df):
    """Vectorised DataFrame-to-dict conversion (avoids iterrows)."""
    df = schedule_df[
        ["expert_id", "slot_start_utc", "slot_end_utc", "assignment", "shift_block"]
    ].copy()
    df["expert_id"] = df["expert_id"].astype(str)
    df["slot_start"] = df["slot_start_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    df["slot_end"] = df["slot_end_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df[["expert_id", "slot_start", "slot_end", "assignment", "shift_block"]].to_dict(
        orient="records"
    )


def _df_to_coverage_dicts(coverage_df):
    if len(coverage_df) == 0:
        return []
    df = coverage_df.copy()
    df["slot_start"] = df["slot_start_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    df["agents_assigned"] = df["agents_assigned"].astype(int)
    df["predicted_demand"] = df["predicted_demand"].astype(int)
    df["coverage_ratio"] = df["coverage_ratio"].round(2)
    return df[["slot_start", "agents_assigned", "predicted_demand", "coverage_ratio"]].to_dict(
        orient="records"
    )


def _df_to_summary_dicts(summary_df):
    if len(summary_df) == 0:
        return []
    df = summary_df.copy()
    df["expert_id"] = df["expert_id"].astype(str)
    df["shift_start"] = df["shift_start"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    df["shift_end"] = df["shift_end"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    df["total_slots"] = df["total_slots"].astype(int)
    df["work_slots"] = df["work_slots"].astype(int)
    df["break_slots"] = df["break_slots"].astype(int)
    df["shift_hours"] = df["shift_hours"].astype(float)
    df["work_hours"] = df["work_hours"].astype(float)
    return df[
        ["expert_id", "shift_block", "shift_start", "shift_end",
         "total_slots", "work_slots", "break_slots", "shift_hours", "work_hours"]
    ].to_dict(orient="records")


def _get_cached_schedule(date, demand_by_slot, max_agents=None):
    """Cache schedule results to avoid recomputing for the same date."""
    cache_key = (str(date), max_agents)
    if cache_key in _schedule_cache:
        return _schedule_cache[cache_key]
    ss = load_shift_scheduler()
    schedule_df = ss.schedule_day(
        target_date=date,
        demand_by_slot=demand_by_slot,
        prefer_high_performers=True,
        max_agents=max_agents,
    )
    _schedule_cache[cache_key] = (ss, schedule_df)
    if len(_schedule_cache) > 32:
        oldest = next(iter(_schedule_cache))
        del _schedule_cache[oldest]
    return ss, schedule_df


@app.get("/api/schedule-with-staffing")
def get_schedule_with_staffing(
    date: str = Query(..., description="Target date (YYYY-MM-DD)"),
    min_sla: float = 0.80,
    max_wait: float = 60.0,
    max_occupancy: float = 0.85,
    max_agents: Optional[int] = Query(
        None, description="Max agents to schedule"
    ),
):
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    try:
        slots = run_pipeline_for_date(date, min_sla, max_wait, max_occupancy)
        demand_by_slot = {}
        for slot in slots:
            h, m = slot["time"].split(":")
            h_local, m = int(h), int(m)
            h_utc = (h_local + 8) % 24
            demand_by_slot[(h_utc, m)] = slot["predicted_calls"]

        ss, schedule_df = _get_cached_schedule(
            date, demand_by_slot, max_agents
        )

        if len(schedule_df) > 0:
            assignments = _df_to_assignment_dicts(schedule_df)
            coverage_df = ss.coverage_report(schedule_df, demand_by_slot)
            coverage = _df_to_coverage_dicts(coverage_df)
            summary_df = ss.agent_shift_summary(schedule_df)
            summary = _df_to_summary_dicts(summary_df)
        else:
            assignments, coverage, summary = [], [], []

        return {
            "staffing": slots,
            "schedule": {
                "date": date,
                "total_active_agents": ss.total_active_agents,
                "assignments": assignments,
                "coverage": coverage,
                "summary": summary,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/schedule")
def get_schedule(
    date: str = Query(..., description="Target date (YYYY-MM-DD)"),
    max_agents: Optional[int] = Query(
        None, description="Maximum agents to schedule"
    ),
):
    try:
        slots = run_pipeline_for_date(date, 0.80, 60.0, 0.85)
        demand_by_slot = {}
        for slot in slots:
            h, m = slot["time"].split(":")
            h_local, m = int(h), int(m)
            h_utc = (h_local + 8) % 24
            demand_by_slot[(h_utc, m)] = slot["predicted_calls"]

        ss, schedule_df = _get_cached_schedule(
            date, demand_by_slot, max_agents
        )

        total_active = ss.total_active_agents

        if len(schedule_df) == 0:
            return {
                "date": date,
                "total_active_agents": total_active,
                "assignments": [],
                "coverage": [],
                "summary": [],
            }

        assignments = _df_to_assignment_dicts(schedule_df)

        coverage_df = ss.coverage_report(schedule_df, demand_by_slot)
        coverage = _df_to_coverage_dicts(coverage_df)

        summary_df = ss.agent_shift_summary(schedule_df)
        summary = _df_to_summary_dicts(summary_df)

        return {
            "date": date,
            "total_active_agents": total_active,
            "assignments": assignments,
            "coverage": coverage,
            "summary": summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/schedule/availability")
def get_agent_availability(
    date: str = Query(..., description="Target date (YYYY-MM-DD)"),
    limit: int = Query(50, ge=1, le=500, description="Max agents to return"),
):
    try:
        ss = load_shift_scheduler()

        target_date = pd.to_datetime(date)
        dow = target_date.dayofweek

        if dow >= 5:
            return {
                "date": date,
                "available_agents": [],
                "total_available": 0,
            }

        if ss._agent_availability is None:
            raise HTTPException(
                status_code=500, detail="Shift scheduler not initialized"
            )

        dow_avail = ss._agent_availability[
            ss._agent_availability["dow"] == dow
        ]

        agent_scores = (
            dow_avail.groupby("expert_id")
            .agg(
                available_hours=("work_frequency", "sum"),
                mean_work_freq=("work_frequency", "mean"),
                mean_occ=("mean_occupancy", "mean"),
            )
            .reset_index()
        )
        total_available = len(agent_scores)

        agent_scores = agent_scores.nlargest(limit, "mean_work_freq")

        if ss._agent_meta is not None:
            agent_scores = agent_scores.merge(
                ss._agent_meta[
                    [
                        "expert_id",
                        "expert_segment",
                        "business_segment",
                        "resolution_rate",
                        "average_handle_time_seconds",
                    ]
                ],
                on="expert_id",
                how="left",
            )

        df = agent_scores.copy()
        df["expert_id"] = df["expert_id"].astype(str)
        df["name"] = "Agent " + df["expert_id"].str[-4:]
        df["segment"] = df.get("expert_segment", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str)
        df["business_segment"] = df.get("business_segment", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str)
        df["available_hours"] = (df["available_hours"] / 2).round(1)
        df["mean_work_freq"] = (df["mean_work_freq"] * 100).round(1)
        df["mean_occupancy"] = df["mean_occ"].round(1)
        df["resolution_rate"] = df.get("resolution_rate", pd.Series(0, index=df.index)).fillna(0).round(1)

        available = df[
            ["expert_id", "name", "segment", "business_segment",
             "available_hours", "mean_work_freq", "mean_occupancy", "resolution_rate"]
        ].to_dict(orient="records")

        return {
            "date": date,
            "day_of_week": [
                "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            ][dow],
            "available_agents": available,
            "total_available": total_available,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/routing/recommend")
def get_routing_recommendations(
    product_group_sku: Optional[str] = Query(
        None,
        description="Product/segment SKU (e.g. SBSEG_US_OL_PRPM_QBO_A_PHN)",
    ),
    channel: Optional[str] = Query(
        None,
        description="Channel: PHONE, CHAT, INBOUND, OUTBOUND",
    ),
    datetime_utc: Optional[str] = Query(
        None,
        description="Target datetime UTC (YYYY-MM-DDTHH:MM) for availability",
    ),
    top_n: int = Query(
        5, ge=1, le=20, description="Number of agents to return"
    ),
):
    try:
        router = load_agent_router()
        analytics = load_agent_analytics()
        recommendations = router.recommend(
            product_group_sku=product_group_sku,
            channel=channel,
            date_utc=datetime_utc,
            top_n=top_n,
            agent_analytics=analytics,
        )
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
