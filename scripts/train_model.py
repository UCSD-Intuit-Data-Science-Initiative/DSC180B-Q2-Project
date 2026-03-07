"""
Offline training script for the RF + GradientBoosting ensemble forecaster.

Trains both models on dataset_1, builds holiday profiles and daily std dev
lookups, then saves everything as a single pkl bundle for the FastAPI server
to load via @lru_cache.

Usage:
    cd <project-root>
    PYTHONPATH=src python scripts/train_model.py

Produces:
    data/models/call_volume_model_bundle.pkl
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
FORECAST_WEEKS = 1

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_1_PATH = PROJECT_ROOT / "data" / "parquet" / "dataset_1_call_related.parquet"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "data" / "models"
MODEL_OUTPUT_PATH = MODEL_OUTPUT_DIR / "call_volume_model_bundle.pkl"

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print(f"Loading data for {FORECAST_WEEKS}-week forecast...")
df = pd.read_parquet(str(DATASET_1_PATH))
df["arrival_time_utc"] = pd.to_datetime(df["arrival_time_utc"])
df = df.sort_values("arrival_time_utc")

# Aggregate to 30-min call volume
df_agg = df.set_index("arrival_time_utc").resample("30min").size().to_frame("Call_Volume")
df_agg = df_agg.asfreq("30min", fill_value=0)

# ==============================================================================
# 1.5. DATA CLEANSING
# ==============================================================================
print("Smoothing known historical anomalies (2025-08-29)...")
outlier_start = "2025-08-29 00:00:00"
outlier_end = "2025-08-29 23:59:59"
bad_data_mask = (df_agg.index >= outlier_start) & (df_agg.index <= outlier_end)
df_agg.loc[bad_data_mask, "Call_Volume"] = np.nan
df_agg["Call_Volume"] = df_agg["Call_Volume"].interpolate(method="time")

# ==============================================================================
# 2. CREATE FUTURE SCHEDULE
# ==============================================================================
last_date = df_agg.index.max()
start_future = last_date + pd.Timedelta(minutes=30)
end_future = start_future + pd.Timedelta(weeks=FORECAST_WEEKS) - pd.Timedelta(minutes=30)

future_dates = pd.date_range(start=start_future, end=end_future, freq="30min")
future_df = pd.DataFrame(index=future_dates)
future_df["Call_Volume"] = np.nan

df_combined = pd.concat([df_agg, future_df])

# ==============================================================================
# 3. DYNAMIC FEATURE ENGINEERING
# ==============================================================================
def create_features(data, weeks_ahead):
    df = data.copy()

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df.index.min(), end=df.index.max())

    major_holidays = [
        "2024-01-01", "2025-01-01", "2026-01-01",
        "2024-11-28", "2025-11-27", "2026-11-26",
        "2024-12-25", "2025-12-25", "2026-12-25",
    ]
    major_holidays = pd.to_datetime(major_holidays)

    df["is_major_holiday"] = df.index.normalize().isin(major_holidays).astype(int)
    df["is_minor_holiday"] = (
        df.index.normalize().isin(holidays) & ~df.index.normalize().isin(major_holidays)
    ).astype(int)

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["weekofyear"] = df.index.isocalendar().week.astype(int)
    df["is_january"] = (df["month"] == 1).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    tax_days = pd.to_datetime(df.index.year.astype(str) + "-04-15")
    df["days_to_tax_day"] = (tax_days - df.index.normalize()).days
    df["is_tax_season"] = ((df["month"] <= 4) & (df["days_to_tax_day"] >= 0)).astype(int)
    df["is_post_tax_drop"] = ((df["days_to_tax_day"] < 0) & (df["days_to_tax_day"] > -31)).astype(int)

    lag_intervals = weeks_ahead * 7 * 48
    rolling_window = weeks_ahead * 7 * 48

    df[f"lag_{weeks_ahead}weeks"] = df["Call_Volume"].shift(lag_intervals)
    df[f"trend_{weeks_ahead}w"] = df["Call_Volume"].shift(lag_intervals).rolling(window=rolling_window).mean()
    df[f"max_{weeks_ahead}w"] = df["Call_Volume"].shift(lag_intervals).rolling(window=rolling_window).max()

    return df


df_features = create_features(df_combined, weeks_ahead=FORECAST_WEEKS)

# ==============================================================================
# 4. PREPARE DATA
# ==============================================================================
lag_feat = f"lag_{FORECAST_WEEKS}weeks"
trend_feat = f"trend_{FORECAST_WEEKS}w"
max_feat = f"max_{FORECAST_WEEKS}w"

train_data = df_features.dropna(subset=["Call_Volume", lag_feat, trend_feat])
predict_data = df_features[df_features["Call_Volume"].isna()].copy()

features = [
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "month", "weekofyear", "is_january",
    "is_minor_holiday", "is_major_holiday",
    "days_to_tax_day", "is_tax_season", "is_post_tax_drop",
    lag_feat, trend_feat, max_feat,
]

print(f"Training set: {len(train_data)} rows, {len(features)} features")
print(f"Prediction set: {len(predict_data)} rows (future slots)")

# ==============================================================================
# 5. TRAIN ENSEMBLE
# ==============================================================================
# --- Model A: Random Forest ---
rf_model = RandomForestRegressor(
    n_estimators=500,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="log2",
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)
print("Training Random Forest (500 trees)...")
rf_model.fit(train_data[features], train_data["Call_Volume"])

# --- Model B: Gradient Boosting ---
gb_model = GradientBoostingRegressor(
    n_estimators=1000,
    min_samples_leaf=5,
    max_depth=7,
    learning_rate=0.01,
    max_features=0.9,
    loss="squared_error",
    random_state=42,
)
print("Training Gradient Boosting (1000 trees)...")
gb_model.fit(train_data[features], train_data["Call_Volume"])

# ==============================================================================
# 6. BUILD HOLIDAY PROFILES (business logic overrides)
# ==============================================================================
print("Building holiday profiles...")
historical_holidays = train_data[train_data["is_major_holiday"] == 1]
holiday_profiles = historical_holidays.groupby(
    [historical_holidays.index.month, historical_holidays.index.day, historical_holidays.index.time]
)["Call_Volume"].mean().to_dict()
print(f"  {len(holiday_profiles)} holiday (month, day, time) entries")

# ==============================================================================
# 7. BUILD DAILY STD DEV LOOKUP (for weekly chart error bars)
# ==============================================================================
print("Building daily std dev lookup...")
df_agg_std = df_agg.copy()
df_agg_std["date_only"] = df_agg_std.index.normalize()
df_agg_std["dow_day"] = df_agg_std.index.dayofweek
daily_totals = df_agg_std.groupby(["dow_day", "date_only"])["Call_Volume"].sum().reset_index()
daily_std_lookup = daily_totals.groupby("dow_day")["Call_Volume"].std().to_dict()
print(f"  {len(daily_std_lookup)} day-of-week entries")

# ==============================================================================
# 7.5. BUILD AHT LOOKUP (per 30-min slot, for Erlang-A optimizer)
# ==============================================================================
print("Building AHT lookup...")
answered = df[df["answered_flag"] == True].copy()
answered["start"] = pd.to_datetime(answered["start_time_utc"])
answered["end"]   = pd.to_datetime(answered["end_time_utc"])
answered["handle_time"] = (answered["end"] - answered["start"]).dt.total_seconds()

# Filter outliers: keep 0 < handle_time < 14400s (4 hours)
answered = answered[(answered["handle_time"] > 0) & (answered["handle_time"] < 14400)]

# Group by (day_of_week, 30-min slot) → mean AHT in seconds
answered["slot"] = answered["start"].dt.floor("30min")
answered["dow"]  = answered["slot"].dt.dayofweek
answered["slot_str"] = answered["slot"].dt.strftime("%H:%M")

aht_lookup = answered.groupby(["dow", "slot_str"])["handle_time"].mean().to_dict()
print(f"  {len(aht_lookup)} (day_of_week, slot) entries")

# ==============================================================================
# 8. SAVE MODEL BUNDLE
# ==============================================================================
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

bundle = {
    "rf_model": rf_model,
    "gb_model": gb_model,
    "features": features,
    "holiday_profiles": holiday_profiles,
    "forecast_weeks": FORECAST_WEEKS,
    "daily_std_lookup": daily_std_lookup,
    "aht_lookup": aht_lookup,
    "call_volume_history": df_agg["Call_Volume"],
    "trained_at": datetime.utcnow().isoformat(),
}

temp_path = MODEL_OUTPUT_PATH.with_suffix(".tmp")
with temp_path.open("wb") as f:
    pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
temp_path.replace(MODEL_OUTPUT_PATH)

file_size_mb = MODEL_OUTPUT_PATH.stat().st_size / (1024 * 1024)
print(f"\nSuccess! Model bundle saved to {MODEL_OUTPUT_PATH} ({file_size_mb:.1f} MB)")
print(f"  RF feature importances (top 5):")
importance = sorted(zip(features, rf_model.feature_importances_), key=lambda x: -x[1])
for name, imp in importance[:5]:
    print(f"    {name}: {imp:.4f}")
