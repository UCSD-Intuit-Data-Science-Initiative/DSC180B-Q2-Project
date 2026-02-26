# Model Card: Dynamic Tax Season Call Volume Forecaster (forecasting_dynamic_weeks.py)

## Model Details
* **Model Type:** Ensemble Regression utilizing Simple Average Blending.
* **Base Models:** * `RandomForestRegressor` from Scikit-Learn utilizing 200 trees, a maximum depth of 25, and sqrt maximum features.
  * `GradientBoostingRegressor` from Scikit-Learn utilizing 1500 trees, a learning rate of 0.01, and a maximum depth of 3.
* **Task:** Time-series forecasting of call center volumes in 30-minute intervals.
* **Output:** Predicted call volume integer count per 30-minute interval for a configurable dynamic forecast window.

## Intended Use
* **Primary Use Case:** Forecasting workforce management needs by predicting future call volumes at 30-minute intervals for a configurable number of weeks into the future.
* **Target Domain:** Customer support call centers heavily influenced by U.S. tax seasons and holidays.

## Training Data
* **Dataset:** 2 years of historical mock data from `mock_intuit_2year_data.csv`.
* **Preprocessing:** Data is aggregated from raw timestamps into 30-minute frequency intervals. A warm-up period is dropped during training to accommodate the dynamic lag features where data is NaN.

## Features
The models rely on engineered features representing time, calendar events, tax logic, and dynamic system memory:
* **Cyclical Time:** Sine and cosine transformations of the hour of the day and day of the week.
* **Calendar Flags:** Includes the month, week of the year, a January indicator, and a US Federal Holiday indicator.
* **Tax Season Logic:** Days remaining until Tax Day (hardcoded to April 15), an active tax season flag (January through April 15), and a post-tax drop flag for the 30 days following tax day.
* **Dynamic Lags & Memory:** The model automatically calculates lag intervals, a 4-week rolling average trend, and a 4-week rolling maximum based on the configured forecast weeks.

## Performance & Metrics
* **Ensemble Strategy:** The final prediction uses a straightforward average of the Random Forest and Gradient Boosting predictions, rounding the result to the nearest integer to represent whole calls. 

## Caveats and Recommendations
* **Rigid Tax Day:** The tax day logic hardcodes the deadline to April 15th every year. If the deadline shifts, the feature engineering logic `days_to_tax_day` will supply incorrect seasonality cues to the models.
* **Dynamic Cold Start Dependency:** Because the model calculates historical lags based on the configured `FORECAST_WEEKS` combined with a 4-week rolling window, generating a forecast requires enough continuous historical data to satisfy this dynamic warm-up period.