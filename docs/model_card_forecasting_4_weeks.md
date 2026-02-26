# Model Card: Tax Season Call Volume Forecaster (forecasting_4_weeks.py)

## Model Details
* **Model Type:** Ensemble Regression (Simple Average Blending).
* **Base Models:** * `RandomForestRegressor`: 200 trees, max depth 25, sqrt max features.
  * `GradientBoostingRegressor`: 1500 trees, learning rate 0.01, max depth 3.
* **Task:** Time-series forecasting of call center volumes.
* **Output:** Predicted call volume integer count per 30-minute interval for a future 28-day window.

## Intended Use
* **Primary Use Case:** Forecasting workforce management (WFM) needs by predicting future call volumes at 30-minute intervals.
* **Target Domain:** Customer support call centers heavily influenced by U.S. tax seasons and holidays (e.g., Intuit).
* **Out-of-Scope Uses:** Predicting call volumes for non-US regions (due to hardcoded US holidays and April 15th tax day) or industries without heavy April seasonality. It is also not designed to forecast real-time anomaly spikes (e.g., unexpected website outages).

## Training Data
* **Dataset:** 2 years of historical mock data (`mock_intuit_2year_data.csv`).
* **Preprocessing:** Data is aggregated from raw timestamps into 30-minute frequency intervals. The first 4 weeks of the dataset are dropped during training to accommodate the 4-week lag feature "warm-up" period.

## Features
The models rely on 14 engineered features representing time, seasonality, and recent system memory:
* **Cyclical Time:** Sine and cosine transformations of hour-of-day and day-of-week.
* **Calendar Flags:** Month, week of year, January flag, and US Federal Holiday flag.
* **Domain-Specific (Tax):** Days remaining until Tax Day (April 15), active tax season flag (Jan-April 15), and a post-tax drop flag (30 days following tax day).
* **Lags & Memory:** 4-week exact lag (`lag_4weeks`), 4-week rolling average trend (`trend_4w`), and 4-week rolling maximum (`max_4w`).

## Performance & Metrics
* **Evaluation Metrics:** *Not explicitly defined in the production script.* Standard development for this type of model typically evaluates using Mean Absolute Error (MAE) or Weighted Mean Absolute Percentage Error (WMAPE) to measure volume discrepancies.
* **Ensemble Strategy:** The final prediction is a 50/50 unweighted average of the Random Forest and Gradient Boosting predictions, rounded to the nearest whole integer. 

## Caveats and Recommendations
* **Rigid Tax Day:** The tax day is hardcoded to April 15th every year. If the IRS pushes back the deadline (as they did in 2020 and 2021), the feature engineering logic `days_to_tax_day` will feed incorrect seasonality cues to the model.
* **Cold Start Dependency:** Because the model relies heavily on a 4-week lookback window, generating a forecast requires at least 4 continuous weeks of reliable historical data preceding the prediction window.