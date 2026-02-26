import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# ==============================================================================
# 1. INPUT: Load your latest historical data
# ==============================================================================
print("Loading historical data...")
df = pd.read_csv("../data/raw/mock_intuit_2year_data.csv")
df["Arrival time"] = pd.to_datetime(df["Arrival time"])
df = df.sort_values("Arrival time")

# Aggregate to 30-min intervals (The model's native language)
df_agg = (
    df.set_index("Arrival time")
    .resample("30min")
    .size()
    .to_frame("Call_Volume")
)
df_agg = df_agg.asfreq("30min", fill_value=0)

# ==============================================================================
# 2. CREATE FUTURE SCHEDULE (The "Blank Page" to fill)
# ==============================================================================
# We want to predict the next 28 days (4 weeks) starting from where data ends
last_date = df_agg.index.max()
start_future = last_date + pd.Timedelta(minutes=30)
end_future = start_future + pd.Timedelta(days=28) - pd.Timedelta(minutes=30)

future_dates = pd.date_range(start=start_future, end=end_future, freq="30min")
future_df = pd.DataFrame(index=future_dates)
future_df["Call_Volume"] = np.nan  # This is what we need to predict!

print(f"Forecasting period: {start_future} to {end_future}")

# Combine History + Future (Crucial for calculating Lags)
# The model needs to look back 4 weeks from the 'Future' rows to see 'History'
df_combined = pd.concat([df_agg, future_df])


# ==============================================================================
# 3. FEATURE ENGINEERING (Apply the Logic we built)
# ==============================================================================
def create_features(data):
    df = data.copy()

    # Calendar & Time
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df.index.min(), end=df.index.max())
    df["is_holiday"] = df.index.normalize().isin(holidays).astype(int)

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["weekofyear"] = df.index.isocalendar().week.astype(int)
    df["is_january"] = (df["month"] == 1).astype(int)

    # Cyclical Encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Tax Season Logic
    tax_days = pd.to_datetime(df.index.year.astype(str) + "-04-15")
    df["days_to_tax_day"] = (tax_days - df.index.normalize()).days
    df["is_tax_season"] = (
        (df["month"] <= 4) & (df["days_to_tax_day"] >= 0)
    ).astype(int)
    df["is_post_tax_drop"] = (
        (df["days_to_tax_day"] < 0) & (df["days_to_tax_day"] > -31)
    ).astype(int)

    # Lags & Trends (The "Memory" of the model)
    # Note: These will be NaN for the first 4 weeks of history, which is fine
    df["lag_4weeks"] = df["Call_Volume"].shift(48 * 28)
    df["trend_4w"] = (
        df["Call_Volume"].shift(48 * 28).rolling(window=48 * 7).mean()
    )
    df["max_4w"] = (
        df["Call_Volume"].shift(48 * 28).rolling(window=48 * 7).max()
    )

    return df


df_features = create_features(df_combined)

# ==============================================================================
# 4. TRAIN & PREDICT
# ==============================================================================
# Train Set: All rows where we actually know the Call_Volume (History)
train_data = df_features.dropna(subset=["Call_Volume"])

# --- THE FIX IS HERE ---
# Drop the first 4 weeks of history where lags are NaN (the "warm-up" period)
train_data = train_data.dropna()
# -----------------------

# Prediction Set: All rows where Call_Volume is NaN (Future)
predict_data = df_features[df_features["Call_Volume"].isna()].copy()

features = [
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month",
    "weekofyear",
    "is_january",
    "is_holiday",
    "days_to_tax_day",
    "is_tax_season",
    "is_post_tax_drop",
    "lag_4weeks",
    "trend_4w",
    "max_4w",
]

# --- A. Train Random Forest ---
print("Training Random Forest...")
rf = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=4,
    max_depth=25,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)
rf.fit(train_data[features], train_data["Call_Volume"])

# --- B. Train XGBoost ---
print("Training XGBoost...")
xgb_model = GradientBoostingRegressor(
    n_estimators=1500,
    learning_rate=0.01,
    max_depth=3,
    min_samples_leaf=3,
    subsample=1.0,
    max_features=0.7,
    random_state=42,
)
xgb_model.fit(train_data[features], train_data["Call_Volume"])

# --- C. Stack Predictions ---
print("Generating Forecast...")
predict_data["pred_rf"] = rf.predict(predict_data[features])
predict_data["pred_xgb"] = xgb_model.predict(predict_data[features])

# Average them (Stacking)
predict_data["Final_Forecast"] = (
    predict_data["pred_rf"] + predict_data["pred_xgb"]
) / 2

# Round to nearest whole call
predict_data["Final_Forecast"] = (
    predict_data["Final_Forecast"].round().astype(int)
)

# ==============================================================================
# 5. OUTPUT: Save to CSV
# ==============================================================================
output_df = predict_data[["Final_Forecast"]]
output_file = "Forecast_2026_Jan.csv"
output_df.to_csv(output_file)

print(f"Success! Forecast saved to {output_file}")
print(output_df.head(10))  # Show user what the output looks like
