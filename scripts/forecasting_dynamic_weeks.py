import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from pandas.tseries.holiday import USFederalHolidayCalendar

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
# Change this number to predict further out (e.g., 4, 8, 12 weeks)
FORECAST_WEEKS = 4 

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print(f"Loading data for {FORECAST_WEEKS}-week forecast...")
df = pd.read_csv('mock_intuit_2year_data.csv')
df['Arrival time'] = pd.to_datetime(df['Arrival time'])
df = df.sort_values('Arrival time')

# Aggregate
df_agg = df.set_index('Arrival time').resample('30min').size().to_frame('Call_Volume')
df_agg = df_agg.asfreq('30min', fill_value=0)

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
    
    # --- Time & Calendar ---
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df.index.min(), end=df.index.max())
    df['is_holiday'] = df.index.normalize().isin(holidays).astype(int)
    
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
    
    # --- Tax Season Logic ---
    tax_days = pd.to_datetime(df.index.year.astype(str) + '-04-15')
    df['days_to_tax_day'] = (tax_days - df.index.normalize()).days
    df['is_tax_season'] = ((df['month'] <= 4) & (df['days_to_tax_day'] >= 0)).astype(int)
    df['is_post_tax_drop'] = ((df['days_to_tax_day'] < 0) & (df['days_to_tax_day'] > -31)).astype(int)

    # --- DYNAMIC LAGS ---
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
# 4. TRAIN & PREDICT
# ==============================================================================
# 1. Get History (Training Data)
train_data = df_features.dropna(subset=['Call_Volume'])

# 2. DROP WARM-UP PERIOD (Crucial Step)
# We remove the first few weeks where the new lag feature is NaN
train_data = train_data.dropna()

# 3. Get Future (Prediction Data)
predict_data = df_features[df_features['Call_Volume'].isna()].copy()

# 4. Define Features List Dynamically
lag_feat = f'lag_{FORECAST_WEEKS}weeks'
trend_feat = f'trend_{FORECAST_WEEKS}w'
max_feat = f'max_{FORECAST_WEEKS}w'

features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
            'month', 'weekofyear', 'is_january', 'is_holiday',
            'days_to_tax_day', 'is_tax_season', 'is_post_tax_drop',
            lag_feat, trend_feat, max_feat]

# 5. Train Models (Stacked)
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=4, max_depth=25, max_features='sqrt', n_jobs=-1, random_state=42)
rf.fit(train_data[features], train_data['Call_Volume'])

print("Training Gradient Boosting...")
gbm_model = GradientBoostingRegressor(n_estimators=1500, learning_rate=0.01, max_depth=3, min_samples_leaf=3, subsample=1.0, max_features=0.7, random_state=42)
gbm_model.fit(train_data[features], train_data['Call_Volume'])

# 6. Generate Forecast
print("Generating Forecast...")
predict_data['pred_rf'] = rf.predict(predict_data[features])
predict_data['pred_gbm'] = gbm_model.predict(predict_data[features])

# Average them
predict_data['Final_Forecast'] = (predict_data['pred_rf'] + predict_data['pred_gbm']) / 2
predict_data['Final_Forecast'] = predict_data['Final_Forecast'].round().astype(int)

# ==============================================================================
# 5. SAVE OUTPUT
# ==============================================================================
output_file = f'Forecast_{FORECAST_WEEKS}Weeks_Ahead.csv'
predict_data[['Final_Forecast']].to_csv(output_file)

print(f"Success! Forecast saved to {output_file}")