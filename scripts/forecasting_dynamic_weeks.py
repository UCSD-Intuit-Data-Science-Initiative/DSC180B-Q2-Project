import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from pandas.tseries.holiday import USFederalHolidayCalendar

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
FORECAST_WEEKS = 1
MODEL_SAVE_PATH = f"call_volume_model_{FORECAST_WEEKS}w_bundle.pkl"

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print(f"Loading data for {FORECAST_WEEKS}-week forecast...")
df = pd.read_parquet("../data/raw/dataset_1_call_related.parquet")
df["arrival_time_utc"] = pd.to_datetime(df["arrival_time_utc"])
df = df.sort_values("arrival_time_utc")

# Aggregate
df_agg = (
    df.set_index("arrival_time_utc")
    .resample("30min")
    .size()
    .to_frame("Call_Volume")
)
df_agg = df_agg.asfreq("30min", fill_value=0)

# ==============================================================================
# 1.5 DATA CLEANSING
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
end_future = (
    start_future
    + pd.Timedelta(weeks=FORECAST_WEEKS)
    - pd.Timedelta(minutes=30)
)

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
        "2024-01-01",
        "2025-01-01",
        "2026-01-01",
        "2024-11-28",
        "2025-11-27",
        "2026-11-26",
        "2024-12-25",
        "2025-12-25",
        "2026-12-25",
    ]
    major_holidays = pd.to_datetime(major_holidays)

    df["is_major_holiday"] = (
        df.index.normalize().isin(major_holidays).astype(int)
    )
    df["is_minor_holiday"] = (
        df.index.normalize().isin(holidays)
        & ~df.index.normalize().isin(major_holidays)
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
    df["is_tax_season"] = (
        (df["month"] <= 4) & (df["days_to_tax_day"] >= 0)
    ).astype(int)
    df["is_post_tax_drop"] = (
        (df["days_to_tax_day"] < 0) & (df["days_to_tax_day"] > -31)
    ).astype(int)

    lag_intervals = weeks_ahead * 7 * 48
    rolling_window = weeks_ahead * 7 * 48

    df[f"lag_{weeks_ahead}weeks"] = df["Call_Volume"].shift(lag_intervals)
    df[f"trend_{weeks_ahead}w"] = (
        df["Call_Volume"]
        .shift(lag_intervals)
        .rolling(window=rolling_window)
        .mean()
    )
    df[f"max_{weeks_ahead}w"] = (
        df["Call_Volume"]
        .shift(lag_intervals)
        .rolling(window=rolling_window)
        .max()
    )

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
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month",
    "weekofyear",
    "is_january",
    "is_minor_holiday",
    "is_major_holiday",
    "days_to_tax_day",
    "is_tax_season",
    "is_post_tax_drop",
    lag_feat,
    trend_feat,
    max_feat,
]

# ==============================================================================
# 5. TRAIN & PREDICT (EXACT HYPERPARAMETERS RESTORED)
# ==============================================================================
# --- Model A: Random Forest ---
rf_model = RandomForestRegressor(
    n_estimators=500,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="log2",
    max_depth=None,
)
print("Training Random Forest...")
rf_model.fit(train_data[features], train_data["Call_Volume"])

# --- Model B: Gradient Boosting Regressor ---
gb_model = GradientBoostingRegressor(
    n_estimators=1000,
    min_samples_leaf=5,
    max_depth=7,
    learning_rate=0.01,
    max_features=0.9,
    loss="squared_error",
)
print("Training Gradient Boosting...")
gb_model.fit(train_data[features], train_data["Call_Volume"])

print("Generating Forecast...")
predict_data["pred_rf"] = rf_model.predict(predict_data[features])
predict_data["pred_gb"] = gb_model.predict(predict_data[features])

predict_data["Final_Forecast"] = (
    predict_data["pred_rf"] + predict_data["pred_gb"]
) / 2

# ==============================================================================
# 6. BUSINESS LOGIC OVERRIDES
# ==============================================================================
print("Applying Business Logic Overrides...")

historical_holidays = train_data[train_data["is_major_holiday"] == 1]
holiday_profiles = (
    historical_holidays.groupby(
        [
            historical_holidays.index.month,
            historical_holidays.index.day,
            historical_holidays.index.time,
        ]
    )["Call_Volume"]
    .mean()
    .to_dict()
)

for idx, row in predict_data[predict_data["is_major_holiday"] == 1].iterrows():
    key = (idx.month, idx.day, idx.time())
    predict_data.loc[idx, "Final_Forecast"] = holiday_profiles.get(key, 15)

predict_data["Final_Forecast"] = (
    predict_data["Final_Forecast"].round().astype(int)
)

# ==============================================================================
# 7. SAVE OUTPUT
# ==============================================================================
output_file = f"Forecast_{FORECAST_WEEKS}Weeks_Ahead.csv"
predict_data[["Final_Forecast"]].to_csv(output_file)

# Save Model Bundle
model_bundle = {
    "rf_model": rf_model,
    "gb_model": gb_model,
    "features": features,
    "holiday_profiles": holiday_profiles,
    "forecast_weeks": FORECAST_WEEKS,
}

with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump(model_bundle, f)

print(f"\nSuccess! Forecast saved to {output_file}")
print(f"Model bundle saved to {MODEL_SAVE_PATH}")
