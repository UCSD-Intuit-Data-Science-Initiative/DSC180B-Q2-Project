"""
Offline training script for the CombinedForecaster.

Trains short-term and long-term LightGBM models on dataset_1, builds holiday
profiles and historical patterns, then saves everything as a single pkl bundle
for the FastAPI server to load via @lru_cache.

Usage:
    cd <project-root>
    PYTHONPATH=src python scripts/train_model.py

Produces:
    data/models/call_volume_model_bundle.pkl
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from main_module.workforce.combined_forecaster import CombinedForecaster

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_1_PATH = (
    PROJECT_ROOT / "data" / "parquet" / "dataset_1_call_related.parquet"
)
MODEL_OUTPUT_DIR = PROJECT_ROOT / "data" / "models"
MODEL_OUTPUT_PATH = MODEL_OUTPUT_DIR / "call_volume_model_bundle.pkl"

print("Training CombinedForecaster...")
print(f"Data path: {DATASET_1_PATH}")

forecaster = CombinedForecaster()
forecaster.train(str(DATASET_1_PATH), train_year=2024, test_year=2025)

MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nBuilding daily std dev lookup...")
raw_data = pd.read_parquet(str(DATASET_1_PATH), columns=["arrival_time_utc"])
raw_data["arrival_time_utc"] = pd.to_datetime(raw_data["arrival_time_utc"])
raw_data["interval_start"] = raw_data["arrival_time_utc"].dt.floor("30min")

df_agg = raw_data.groupby("interval_start").size().to_frame("Call_Volume")
df_agg = df_agg.asfreq("30min", fill_value=0)

df_agg["date_only"] = pd.DatetimeIndex(df_agg.index).normalize()
df_agg["dow_day"] = pd.DatetimeIndex(df_agg.index).dayofweek
daily_totals = (
    df_agg.groupby(["dow_day", "date_only"])["Call_Volume"].sum().reset_index()
)
daily_std_lookup = (
    daily_totals.groupby("dow_day")["Call_Volume"].std().to_dict()
)
print(f"  {len(daily_std_lookup)} day-of-week entries")

print("Building AHT lookup...")
answered = pd.read_parquet(
    str(DATASET_1_PATH),
    columns=["answered_flag", "start_time_utc", "end_time_utc"],
)
answered = answered[answered["answered_flag"]].copy()
answered["start"] = pd.to_datetime(answered["start_time_utc"])
answered["end"] = pd.to_datetime(answered["end_time_utc"])
answered["handle_time"] = (
    answered["end"] - answered["start"]
).dt.total_seconds()

answered = answered[
    (answered["handle_time"] > 0) & (answered["handle_time"] < 14400)
]

answered["slot"] = answered["start"].dt.floor("30min")
answered["dow"] = answered["slot"].dt.dayofweek
answered["slot_str"] = answered["slot"].dt.strftime("%H:%M")

aht_lookup = (
    answered.groupby(["dow", "slot_str"])["handle_time"].mean().to_dict()
)
print(f"  {len(aht_lookup)} (day_of_week, slot) entries")

print("\nSaving model bundle...")
forecaster.daily_std_lookup = daily_std_lookup
forecaster.aht_lookup = aht_lookup
forecaster.trained_at = datetime.utcnow().isoformat()

forecaster.save_model(str(MODEL_OUTPUT_PATH))

file_size_mb = MODEL_OUTPUT_PATH.stat().st_size / (1024 * 1024)
print(f"\nSuccess! Model saved to {MODEL_OUTPUT_PATH} ({file_size_mb:.1f} MB)")

if forecaster.feature_importance.get("short_term"):
    print("  Short-term feature importances (top 5):")
    imp = sorted(
        forecaster.feature_importance["short_term"].items(),
        key=lambda x: -x[1],
    )
    for name, val in imp[:5]:
        print(f"    {name}: {val:.4f}")

if forecaster.feature_importance.get("long_term"):
    print("  Long-term feature importances (top 5):")
    imp = sorted(
        forecaster.feature_importance["long_term"].items(), key=lambda x: -x[1]
    )
    for name, val in imp[:5]:
        print(f"    {name}: {val:.4f}")
