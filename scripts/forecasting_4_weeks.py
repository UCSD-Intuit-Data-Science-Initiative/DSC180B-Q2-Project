import numpy as np
import pandas as pd
import pickle
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# ==============================================================================
# 1. INPUT: Load your latest historical data
# ==============================================================================
print("Loading historical data...")
df = pd.read_parquet('../data/raw/dataset_1_call_related.parquet')
df['arrival_time_utc'] = pd.to_datetime(df['arrival_time_utc'])
df = df.sort_values('arrival_time_utc')

# Aggregate to 30-min intervals
df_agg = (
    df.set_index('arrival_time_utc')
    .resample('30min')
    .size()
    .to_frame('Call_Volume')
)
df_agg = df_agg.asfreq('30min', fill_value=0)

# ==============================================================================
# 1.5 DATA CLEANSING (Smooth known historical anomalies before training)
# ==============================================================================
print("Smoothing known historical anomalies (2025-08-29)...")
outlier_start = '2025-08-29 00:00:00'
outlier_end = '2025-08-29 23:59:59'
bad_data_mask = (df_agg.index >= outlier_start) & (df_agg.index <= outlier_end)

# Replace the spiked volume with NaN, then interpolate using surrounding normal days
df_agg.loc[bad_data_mask, 'Call_Volume'] = np.nan
df_agg['Call_Volume'] = df_agg['Call_Volume'].interpolate(method='time')

# ==============================================================================
# 2. CREATE FUTURE SCHEDULE (The "Blank Page" to fill)
# ==============================================================================
last_date = df_agg.index.max()
start_future = last_date + pd.Timedelta(minutes=30)
end_future = start_future + pd.Timedelta(days=28) - pd.Timedelta(minutes=30)

future_dates = pd.date_range(start=start_future, end=end_future, freq='30min')
future_df = pd.DataFrame(index=future_dates)
future_df['Call_Volume'] = np.nan  

print(f"Forecasting period: {start_future} to {end_future}")

df_combined = pd.concat([df_agg, future_df])

# ==============================================================================
# 3. FEATURE ENGINEERING 
# ==============================================================================
def create_features(data):
    df = data.copy()

    # --- A. Calendar & Time ---
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df.index.min(), end=df.index.max())
    
    major_holidays = [
        '2024-01-01', '2025-01-01', '2026-01-01', 
        '2024-11-28', '2025-11-27', '2026-11-26', 
        '2024-12-25', '2025-12-25', '2026-12-25'  
    ]
    major_holidays = pd.to_datetime(major_holidays)
    
    df['is_major_holiday'] = df.index.normalize().isin(major_holidays).astype(int)
    df['is_minor_holiday'] = (
        df.index.normalize().isin(holidays) & ~df.index.normalize().isin(major_holidays)
    ).astype(int)

    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['is_january'] = (df['month'] == 1).astype(int)

    # --- B. Cyclical Encoding ---
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # --- C. Tax Season Logic ---
    tax_days = pd.to_datetime(df.index.year.astype(str) + '-04-15')
    df['days_to_tax_day'] = (tax_days - df.index.normalize()).days
    df['is_tax_season'] = (
        (df['month'] <= 4) & (df['days_to_tax_day'] >= 0)
    ).astype(int)
    df['is_post_tax_drop'] = (
        (df['days_to_tax_day'] < 0) & (df['days_to_tax_day'] > -31)
    ).astype(int)

    # --- D. Lags & Trends ---
    df['lag_4weeks'] = df['Call_Volume'].shift(48 * 28)
    df['trend_4w'] = df['Call_Volume'].shift(48 * 28).rolling(window=48 * 7).mean()
    df['max_4w'] = df['Call_Volume'].shift(48 * 28).rolling(window=48 * 7).max()

    return df

df_features = create_features(df_combined)

# ==============================================================================
# 4. TRAIN MODELS (EXACT HYPERPARAMETERS RESTORED)
# ==============================================================================
train_data = df_features.dropna(subset=['Call_Volume', 'lag_4weeks', 'trend_4w'])
predict_data = df_features[df_features['Call_Volume'].isna()].copy()

features = [
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'month', 'weekofyear', 'is_january', 
    'is_minor_holiday', 'is_major_holiday',
    'days_to_tax_day', 'is_tax_season', 'is_post_tax_drop',
    'lag_4weeks', 'trend_4w', 'max_4w',
]

# --- Model A: Random Forest ---
rf_model = RandomForestRegressor(
    n_estimators=500, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    max_features='log2',
    max_depth=None
)
print("Training Random Forest...")
rf_model.fit(train_data[features], train_data['Call_Volume'])

# --- Model B: Gradient Boosting Regressor ---
gb_model = GradientBoostingRegressor(
    n_estimators=1000,
    min_samples_leaf=5,
    max_depth=7,
    learning_rate=0.01,
    max_features=0.9,
    loss='squared_error'
)
print("Training Gradient Boosting...")
gb_model.fit(train_data[features], train_data['Call_Volume'])

# ==============================================================================
# 5. PREDICT & APPLY OVERRIDES
# ==============================================================================
print("Generating Forecast...")
predict_data['pred_rf'] = rf_model.predict(predict_data[features])
predict_data['pred_gb'] = gb_model.predict(predict_data[features])

# Stack predictions
predict_data['Final_Forecast'] = (predict_data['pred_rf'] + predict_data['pred_gb']) / 2

print("Applying Business Logic Overrides...")
# Grab historical data where it was a Major Holiday to build profiles
historical_holidays = train_data[train_data['is_major_holiday'] == 1]
holiday_profiles = historical_holidays.groupby(
    [historical_holidays.index.month, historical_holidays.index.day, historical_holidays.index.time]
)['Call_Volume'].mean().to_dict()

# Overwrite ML predictions if a future day is a closed holiday
for idx, row in predict_data[predict_data['is_major_holiday'] == 1].iterrows():
    key = (idx.month, idx.day, idx.time())
    predict_data.loc[idx, 'Final_Forecast'] = holiday_profiles.get(key, 15)

# Round to nearest whole call
predict_data['Final_Forecast'] = predict_data['Final_Forecast'].round().astype(int)

# ==============================================================================
# 6. OUTPUT & SAVE MODEL
# ==============================================================================
output_df = predict_data[['Final_Forecast']]
output_file = "Forecast_Next_28_Days.csv"
output_df.to_csv(output_file)

# Save Models and Metadata
model_bundle = {
    'rf_model': rf_model,
    'gb_model': gb_model,
    'features': features,
    'holiday_profiles': holiday_profiles
}

with open('call_volume_model_bundle.pkl', 'wb') as f:
    pickle.dump(model_bundle, f)

print(f"\nSuccess! Forecast saved to {output_file}")
print("Model bundle saved to 'call_volume_model_bundle.pkl'")