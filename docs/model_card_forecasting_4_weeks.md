# Model Card: Tax Season Call Volume Forecaster (Standard 4-Week Horizon)

## Model Details
* **Model Type:** Ensemble Regression (Simple Average Blending) with Post-Processing Business Logic.
* **Base Models:** * `RandomForestRegressor`: 500 trees, unconstrained max depth (`None`), `log2` max features, minimum samples split of 2, minimum samples leaf of 1.
  * `GradientBoostingRegressor`: 1000 trees, learning rate 0.01, max depth 7, maximum features of 0.9, minimum samples leaf of 5, optimized for `squared_error` loss.
* **Task:** Time-series forecasting of call center volumes.
* **Output:** Predicted call volume integer count per 30-minute interval for a future 28-day window. The models and historical holiday profiles are exported together as a `.pkl` bundle for streamlined inference.

## Intended Use
* **Primary Use Case:** Forecasting workforce management (WFM) needs by predicting future call volumes at 30-minute intervals.
* **Target Domain:** Customer support call centers heavily influenced by U.S. tax seasons and holidays.
* **Out-of-Scope Uses:** Predicting call volumes for non-US regions (due to hardcoded US holidays and April 15th tax day) or industries without heavy April seasonality. It is not designed to forecast real-time anomaly spikes (e.g., unexpected website outages).

## Training Data
* **Dataset:** Historical parquet data (`dataset_1_call_related.parquet`).
* **Preprocessing & Cleansing:** * Data is aggregated from raw timestamps into 30-minute frequency intervals. 
  * **Known Anomaly Handling:** A massive historical data corruption/outage spanning the entirety of **August 29, 2025** (00:00:00 to 23:59:59) is explicitly masked with `NaN` and smoothed using time-based interpolation to prevent training bias.
  * The first 4 weeks of the dataset are dropped during training to accommodate the 4-week lag feature "warm-up" period.

## Features
The models rely on engineered features representing time, seasonality, and recent system memory:
* **Cyclical Time:** Sine and cosine transformations of hour-of-day and day-of-week.
* **Calendar Flags:** Month, week of year, January flag. 
* **Holiday Split:** Calendar events are divided into `is_minor_holiday` (standard federal holidays) and `is_major_holiday` (center closed/near-zero volume days like Christmas and Thanksgiving).
* **Domain-Specific (Tax):** Days remaining until Tax Day (April 15), active tax season flag (Jan-April 15), and a post-tax drop flag (30 days following tax day).
* **Lags & Memory:** 4-week exact lag (`lag_4weeks`), 4-week rolling average trend (`trend_4w`), and 4-week rolling maximum (`max_4w`).

## Performance & Metrics
Model performance was evaluated using a dual-metric approach during development to separate pure machine learning accuracy from business-logic application.

* **Version 1: Full Year (Includes Holiday Overrides & 2025-08-29 Outage)**
  * **MAE:** 105.56 calls
  * **RMSE:** 237.74 calls
  * **WMAPE:** 11.57%
  * **R-Squared:** 0.966

* **Version 2: Pure ML Accuracy (Normal Days Only, 2025-08-29 Excluded)**
  * **MAE:** 103.37 calls
  * **RMSE:** 206.68 calls
  * **WMAPE:** 11.34%
  * **R-Squared:** 0.974

* **Ensemble Strategy:** The baseline prediction is a 50/50 unweighted average of the Random Forest and Gradient Boosting predictions. 
* **Business Logic Overrides:** If the forecast falls on an `is_major_holiday`, the ML prediction is discarded and overwritten with the historical average for that specific date and time interval from the training data. The final output is rounded to the nearest integer.

## Caveats and Recommendations
* **Rigid Tax Day:** The tax day is hardcoded to April 15th every year. If the IRS pushes back the deadline, the feature engineering logic `days_to_tax_day` will feed incorrect seasonality cues to the model.
* **Cold Start Dependency:** Because the model relies heavily on a 4-week lookback window, generating a forecast requires at least 4 continuous weeks of reliable historical data preceding the prediction window.

---

# Model Card: Dynamic Tax Season Call Volume Forecaster (Configurable Horizon)

## Model Details
* **Model Type:** Ensemble Regression utilizing Simple Average Blending with Post-Processing Business Logic.
* **Base Models:**
  * `RandomForestRegressor`: 500 trees, unconstrained max depth (`None`), `log2` max features, minimum samples split of 2, minimum samples leaf of 1.
  * `GradientBoostingRegressor`: 1000 trees, learning rate 0.01, max depth 7, maximum features of 0.9, minimum samples leaf of 5, optimized for `squared_error` loss.
* **Task:** Time-series forecasting of call center volumes in 30-minute intervals.
* **Output:** Predicted call volume integer count per 30-minute interval for a configurable dynamic forecast window. Exported alongside features and holiday profiles as a serialized `.pkl` bundle.

## Intended Use
* **Primary Use Case:** Forecasting workforce management needs by predicting future call volumes at 30-minute intervals for a configurable number of weeks into the future (e.g., 1, 4, 8, or 12 weeks ahead).
* **Target Domain:** Customer support call centers heavily influenced by U.S. tax seasons and holidays.

## Training Data
* **Dataset:** Historical parquet data (`dataset_1_call_related.parquet`).
* **Preprocessing & Cleansing:** * Data is aggregated from raw timestamps into 30-minute intervals.
  * **Known Anomaly Handling:** The entire day of **August 29, 2025** (00:00:00 to 23:59:59) is masked with `NaN` and interpolated to smooth out a outlier, preventing the model from learning the anomaly as a recurring pattern.
  * A dynamic warm-up period is dropped during training to accommodate the dynamic lag features where historical data is initially unavailable.

## Features
The models rely on engineered features representing time, calendar events, tax logic, and dynamic system memory:
* **Cyclical Time:** Sine and cosine transformations of the hour of the day and day of the week.
* **Calendar Flags:** Includes the month, week of the year, and a January indicator.
* **Holiday Split:** Calendar events are divided into `is_minor_holiday` and `is_major_holiday` (center closures).
* **Tax Season Logic:** Days remaining until Tax Day (hardcoded to April 15), an active tax season flag (January through April 15), and a post-tax drop flag for the 30 days following tax day.
* **Dynamic Lags & Memory:** The model automatically calculates lag intervals (`lag_{X}weeks`), a rolling average trend (`trend_{X}w`), and a rolling maximum (`max_{X}w`) based strictly on the user-configured `FORECAST_WEEKS` variable.

## Performance & Metrics
* **Evaluation Baseline:** See the 4-Week Horizon Model Card for baseline performance expectations (11.34% WMAPE on normal days). Dynamic models forecasting further into the future (e.g., 8 or 12 weeks) will naturally experience a degradation in accuracy compared to the 4-week baseline due to the extended lag horizon.
* **Ensemble Strategy:** The final prediction uses a straightforward 50/50 average of the Random Forest and Gradient Boosting predictions.
* **Business Logic Overrides:** Major holidays trigger a system override where the machine learning prediction is replaced by an exact historical average for that specific month, day, and time interval. All final predictions are rounded to the nearest whole call.

## Caveats and Recommendations
* **Rigid Tax Day:** The tax day logic hardcodes the deadline to April 15th every year. If the deadline shifts, the feature engineering logic `days_to_tax_day` will supply incorrect seasonality cues to the models.
* **Dynamic Cold Start Dependency:** Because the model calculates historical lags based on the configured `FORECAST_WEEKS`, generating a forecast requires enough continuous historical data to satisfy this specific dynamic warm-up period (e.g., an 8-week forecast requires at least 8 weeks of historical data to generate the first prediction).