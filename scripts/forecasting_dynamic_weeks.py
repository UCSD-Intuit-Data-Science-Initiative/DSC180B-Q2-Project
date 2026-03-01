import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from pandas.tseries.holiday import USFederalHolidayCalendar

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
# Change this number (e.g., 4, 8, 12 weeks)
FORECAST_WEEKS = 1

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print(f"Loading data for {FORECAST_WEEKS}-week forecast...")
df = pd.read_parquet('../data/raw/dataset_1_call_related.parquet')
df['arrival_time_utc'] = pd.to_datetime(df['arrival_time_utc'])
df = df.sort_values('arrival_time_utc')

# Aggregate
df_agg = df.set_index('arrival_time_utc').resample('30min').size().to_frame('Call_Volume')
df_agg = df_agg.asfreq('30min', fill_value=0)

# ==============================================================================
# 1.5 DATA CLEANSING (Smooth known historical anomalies before training)
# ==============================================================================
print("Smoothing known historical anomalies...")
outage_start = '2025-08-29 12:00:00'
outage_end = '2025-08-29 16:00:00'
bad_data_mask = (df_agg.index >= outage_start) & (df_agg.index <= outage_end)

# Replace the spiked volume with NaN, then interpolate using surrounding normal days
df_agg.loc[bad_data_mask, 'Call_Volume'] = np.nan
df_agg['Call_Volume'] = df_agg['Call_Volume'].interpolate(method='time')

# ==============================================================================
# 2. CREATE FUTURE SCHEDULE
# ==============================================================================
last_date = df_agg.index.max()
start_future = last_date + pd.Timedelta(minutes=30)
end_future = start_future + pd.Timedelta(weeks=FORECAST_WEEKS) - pd.Timedelta(minutes=30)

future_dates = pd.date_range(start=start_future, end=end_future, freq='30min')
future_df = pd.DataFrame(index=future_dates)
future_df['Call_Volume'] = np.nan 

# Combine
df_combined = pd.concat([df_agg, future_df])

# ==============================================================================
# 3. DYNAMIC FEATURE ENGINEERING
# ==============================================================================
def create_features(data, weeks_ahead):
    df = data.copy()
    
    # --- A. Time & Calendar (Upgraded Split) ---
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df.index.min(), end=df.index.max())
    
    # Define Major Holidays (Center closed/ghost town)
    major_holidays = [
        '2024-01-01', '2025-01-01', '2026-01-01', # New Years
        '2024-11-28', '2025-11-27', '2026-11-26', # Thanksgiving
        '2024-12-25', '2025-12-25', '2026-12-25'  # Christmas
    ]
    major_holidays = pd.to_datetime(major_holidays)
    
    # Split into distinct buckets
    df['is_major_holiday'] = df.index.normalize().isin(major_holidays).astype(int)
    df['is_minor_holiday'] = (
        df.index.normalize().isin(holidays) & ~df.index.normalize().isin(major_holidays)
    ).astype(int)
    
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['is_january'] = (df['month'] == 1).astype(int)
    
    # Cyclical Encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # --- B. Tax Season Logic ---
    tax_days = pd.to_datetime(df.index.year.astype(str) + '-04-15')
    df['days_to_tax_day'] = (tax_days - df.index.normalize()).days
    df['is_tax_season'] = ((df['month'] <= 4) & (df['days_to_tax_day'] >= 0)).astype(int)
    df['is_post_tax_drop'] = ((df['days_to_tax_day'] < 0) & (df['days_to_tax_day'] > -31)).astype(int)

    # --- C. DYNAMIC LAGS ---
    # Automatically calculates the correct lag based on your configuration
    lag_intervals = weeks_ahead * 7 * 48
    rolling_window = weeks_ahead * 7 * 48
    
    # Dynamic Column Names
    df[f'lag_{weeks_ahead}weeks'] = df['Call_Volume'].shift(lag_intervals)
    df[f'trend_{weeks_ahead}w'] = df['Call_Volume'].shift(lag_intervals).rolling(window=rolling_window).mean()
    df[f'max_{weeks_ahead}w'] = df['Call_Volume'].shift(lag_intervals).rolling(window=rolling_window).max()
    
    return df

df_features = create_features(df_combined, weeks_ahead=FORECAST_WEEKS)

# ==============================================================================
# 4. PREPARE DATA
# ==============================================================================
# 1. Get History (Training Data)
train_data = df_features.dropna(subset=['Call_Volume'])

# 2. DROP WARM-UP PERIOD (Crucial Step)
train_data = train_data.dropna()

# 3. Get Future (Prediction Data)
predict_data = df_features[df_features['Call_Volume'].isna()].copy()

# 4. Define Features List Dynamically
lag_feat = f'lag_{FORECAST_WEEKS}weeks'
trend_feat = f'trend_{FORECAST_WEEKS}w'
max_feat = f'max_{FORECAST_WEEKS}w'

features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
            'month', 'weekofyear', 'is_january', 
            'is_minor_holiday', 'is_major_holiday',
            'days_to_tax_day', 'is_tax_season', 'is_post_tax_drop',
            lag_feat, trend_feat, max_feat]

# ==============================================================================
# 5. TRAIN & PREDICT
# ==============================================================================
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=4, max_depth=25, max_features='sqrt', n_jobs=-1, random_state=42)
rf.fit(train_data[features], train_data['Call_Volume'])

print("Training Gradient Boosting...")
gbm_model = GradientBoostingRegressor(n_estimators=1500, learning_rate=0.01, max_depth=3, min_samples_leaf=3, subsample=1.0, max_features=0.7, random_state=42)
gbm_model.fit(train_data[features], train_data['Call_Volume'])

print("Generating Forecast...")
predict_data['pred_rf'] = rf.predict(predict_data[features])
predict_data['pred_gbm'] = gbm_model.predict(predict_data[features])

# Average them
predict_data['Final_Forecast'] = (predict_data['pred_rf'] + predict_data['pred_gbm']) / 2

# ==============================================================================
# 6. BUSINESS LOGIC OVERRIDES
# ==============================================================================
print("Applying Business Logic Overrides...")

# Grab historical data where it was a Major Holiday to build profiles
historical_holidays = train_data[train_data['is_major_holiday'] == 1]
holiday_profiles = historical_holidays.groupby(
    [historical_holidays.index.month, historical_holidays.index.day, historical_holidays.index.time]
)['Call_Volume'].mean().to_dict()

# Overwrite ML predictions if a future day is a closed holiday
for idx, row in predict_data[predict_data['is_major_holiday'] == 1].iterrows():
    key = (idx.month, idx.day, idx.time())
    if key in holiday_profiles:
        predict_data.loc[idx, 'Final_Forecast'] = holiday_profiles[key]
    else:
        predict_data.loc[idx, 'Final_Forecast'] = 15 # Safety fallback

# Round to nearest whole call
predict_data['Final_Forecast'] = predict_data['Final_Forecast'].round().astype(int)

# ==============================================================================
# 7. SAVE OUTPUT
# ==============================================================================
output_file = f'Forecast_{FORECAST_WEEKS}Weeks_Ahead.csv'
predict_data[['Final_Forecast']].to_csv(output_file)

print(f"\nSuccess! Forecast saved to {output_file}")
print("Preview of first 5 intervals:")
print(predict_data[['Final_Forecast']].head())
