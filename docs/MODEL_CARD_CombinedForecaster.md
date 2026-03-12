---
layout: default
title: Model Card
parent: Methods
nav_order: 1
---

# Model Card: Combined Demand Forecaster

## Model Summary

### Overview

| Property | Value |
|---|---|
| **Model Name** | CombinedForecaster |
| **Version** | 1.0 |
| **Type** | Regression (Multi-Horizon Time Series Forecasting) |
| **Architecture** | Dual LightGBM (Short-term single model + Long-term 3-model ensemble) with anomaly smoothing and holiday overrides |
| **File** | `src/main_module/workforce/combined_forecaster.py` |
| **Saved Model** | `scripts/combined_forecast_model.pkl` |

### Description

The Combined Demand Forecaster unifies the best elements of two predecessor models — the **HybridForecaster** (multi-dataset, multi-horizon LightGBM architecture) and the **Dynamic Weeks Forecaster** (anomaly smoothing, major/minor holiday distinction, holiday profile overrides, and tax-cycle features). It predicts 30-minute interval call volume across two horizons: a **short-term model** (< 7 days ahead) using recent lags and operational features, and a **long-term 3-model ensemble** (≥ 7 days ahead) using historical patterns and year-over-year indicators. On major holidays, ML predictions are bypassed in favor of historical holiday profiles for more reliable estimates.

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   COMBINED FORECASTER v1                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │               DATA PREPROCESSING                           │  │
│  │  • Anomaly smoothing (known outages interpolated)          │  │
│  │  • Multi-dataset merge (datasets 1, 3, 4)                 │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              PREDICTION ROUTING                            │  │
│  │  1. Major holiday? → Holiday profile lookup (bypass ML)    │  │
│  │  2. Horizon < 7 days? → Short-Term Model                  │  │
│  │  3. Horizon ≥ 7 days? → Long-Term Ensemble                │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────┐     ┌──────────────────────────────┐  │
│  │  SHORT-TERM MODEL    │     │  LONG-TERM ENSEMBLE          │  │
│  │  (1× LGBMRegressor)  │     │  (3× LGBMRegressor, avg)    │  │
│  ├──────────────────────┤     ├──────────────────────────────┤  │
│  │  • 800 estimators    │     │  Model A: 1500 est, lr=0.015 │  │
│  │  • lr = 0.03         │     │  Model B: 1500 est, lr=0.015 │  │
│  │  • 127 leaves        │     │  Model C: 1500 est, lr=0.02  │  │
│  │  • Early stopping    │     │  • All with early stopping   │  │
│  │  • Linear recency    │     │  • Quadratic recency weights │  │
│  │    weights            │     │  • L1=1.0, L2=2.0 reg       │  │
│  │                      │     │  • Predictions averaged      │  │
│  │  (55 features)       │     │  (45 features)               │  │
│  └──────────────────────┘     └──────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │             HISTORICAL PATTERN LOOKUP                      │  │
│  │  Pre-computed: dow×hour, month×dow×hour, week-of-year,    │  │
│  │  quarter×dow, time-slot means/stds, YoY patterns          │  │
│  │  + Major holiday profiles (48 intervals per holiday)      │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### What It Combines

| Feature | Source: HybridForecaster | Source: Dynamic Weeks |
|---|---|---|
| Multi-horizon (ST + LT) | ✓ | |
| LightGBM + early stopping | ✓ | |
| 3-model LT ensemble | ✓ | |
| Multi-dataset integration (3 parquets) | ✓ | |
| Recency-weighted training | ✓ | |
| Channel mix features | ✓ | |
| Operational metric features | ✓ | |
| Year-aligned train/test split | ✓ | |
| RobustScaler | ✓ | |
| Anomaly smoothing (outage interpolation) | | ✓ |
| Major vs. minor holiday distinction | | ✓ |
| Holiday profile overrides at inference | | ✓ |
| `is_january` feature | | ✓ |
| `is_post_tax_drop` feature | | ✓ |

### Inputs and Outputs

**Input:**
- Historical call center data (Parquet format, from `dataset_1_call_related.parquet`)
- Supplementary operational data (`dataset_3_historical_outcomes.parquet`, `dataset_4_expert_state_interval.parquet`)
- Target datetime for prediction
- Forecast horizon (automatic routing)

**Short-Term Model Features (55 total):**

| Category | Features | Count |
|---|---|---|
| Temporal | hour, minute, day_of_week, day_of_month, month, time_slot, week_of_year, day_of_year | 8 |
| Tax/Holiday | is_holiday, is_major_holiday, is_january, days_to_tax_deadline, tax_urgency, is_post_tax_drop | 6 |
| Cyclical Encoding | hour_sin, hour_cos, dow_sin, month_sin, month_cos | 5 |
| Lag Features | lag_1, lag_2, lag_4, lag_48, lag_336, lag_672, lag_same_time_yesterday, lag_same_time_last_week | 8 |
| Difference Features | diff_1, diff_48, diff_336 | 3 |
| Rolling Statistics | rolling_mean_4/12/48/336, rolling_std_4/48, rolling_max_4 | 7 |
| EWM Features | ewm_mean_12, ewm_mean_48 | 2 |
| Trend Features | hourly_trend, daily_trend | 2 |
| Advanced | volatility_ratio, momentum | 2 |
| Channel Mix | inbound_ratio, chat_ratio, callback_ratio | 3 |
| Operational Lags | lag_transfer_rate, lag_fcr_rate, lag_mean_hold, lag_active_experts, lag_mean_occupancy, lag_total_avail | 6 |
| Operational Rolling | rolling_experts_48, rolling_occupancy_48 | 2 |
| Year-over-Year | yoy_same_dow_hour_mean | 1 |

**Long-Term Model Features (45 total):**

| Category | Features | Count |
|---|---|---|
| Temporal | hour, day_of_week, day_of_month, month, time_slot, week_of_year, day_of_year | 7 |
| Tax/Holiday | is_holiday, is_major_holiday, is_january, days_to_tax_deadline, tax_urgency, is_post_tax_drop | 6 |
| Cyclical Encoding | hour_cos, dow_sin, month_sin, month_cos | 4 |
| Historical Aggregates | hist_dow_hour_mean/std/median, hist_month_dow_hour_mean/std, hist_month_mean, hist_time_slot_mean, hist_week_of_year_mean, hist_quarter_dow_mean | 9 |
| Long Rolling | rolling_mean_336/672, ewm_mean_336/672 | 4 |
| Channel Mix | inbound_ratio, chat_ratio, callback_ratio | 3 |
| Historical Operational | hist_transfer_rate, hist_fcr_rate, hist_mean_hold, hist_mean_experts, hist_mean_occupancy | 5 |
| Year-over-Year | yoy_same_dow_hour_mean, yoy_same_week_mean | 2 |
| Recent Window | recent_quarter_mean, recent_month_mean, hist_recent_dow_hour_mean | 3 |
| Slot Aggregates | hist_dow_time_slot_mean, hist_month_time_slot_mean | 2 |

**Output:**
- Predicted call count (integer, clipped ≥ 0) for a 30-minute interval

---

## Model Usage and Limitations

### Intended Usage

- **Primary Use:** Multi-horizon call volume forecasting for Intuit QuickBooks / SBSEG support
- **Users:** Call center managers, workforce planners, capacity analysts
- **Applications:**
  - Short-term scheduling (1–7 days ahead)
  - Long-term capacity planning (1–4+ weeks ahead)
  - Seasonal workforce budgeting (tax season preparation)
  - Integration with CallCenterEmulator and SupplyOptimizer for staffing recommendations

### Benefits Over Predecessor Models

- **Anomaly Robustness:** Known data outages (e.g., 2025-08-29) are automatically smoothed via interpolation, preventing the model from training on corrupted intervals
- **Holiday Accuracy:** Major holidays (New Year's, Thanksgiving, Christmas) use historical profile lookup instead of ML prediction, which is more reliable for these rare, extreme-pattern days
- **Richer Calendar Signals:** `is_major_holiday`, `is_january`, and `is_post_tax_drop` capture domain-specific seasonal patterns that the pure HybridForecaster lacked
- **Lower Extreme Errors:** Anomaly smoothing and holiday overrides produce a lower RMSE than the HybridForecaster, meaning fewer large prediction misses
- **All HybridForecaster Strengths Retained:** Multi-dataset integration, dual-horizon architecture, recency weighting, LightGBM ensemble, operational features

### Limitations

- **Year-over-Year Drift:** A 5–15% volume decline was observed between 2024 and 2025; recency weighting mitigates but does not fully eliminate this
- **Business Hours:** Assumes UTC timestamps with Pacific Time business hours (UTC 13:00–01:00, Mon–Fri)
- **Training Data Requirement:** Requires data spanning at least two years for YoY features
- **Long-Term Accuracy:** WMAPE of ~13% for ≥7-day forecasts reflects inherent difficulty of long-horizon prediction
- **Domain Specific:** Optimized for Intuit QB/SBSEG call patterns; requires retraining for other domains
- **Known Anomalies List:** The `_KNOWN_ANOMALIES` list must be manually updated when new outages are identified

### Out-of-Scope Uses

- Sub-interval predictions (less than 30 minutes)
- Individual call outcome or duration prediction
- Non-call-center demand forecasting without retraining
- Real-time anomaly detection

---

## Evaluation

### Performance Metrics

**Test Set Performance (Train: Jan–Oct 2024, Test: Jan–Oct 2025):**

| Metric | Short-Term (< 7 days) | Long-Term (≥ 7 days) |
|---|---|---|
| **MAE** | 28.56 calls | 119.22 calls |
| **RMSE** | 58.96 calls | 220.86 calls |
| **R²** | 0.9979 | 0.9705 |
| **WMAPE** | 3.11% | 13.00% |

### Head-to-Head Comparison (Same Test Set)

| Model | MAE | RMSE | R² | WMAPE | Features |
|---|---|---|---|---|---|
| **Combined (ST)** | 28.56 | **58.96** | **0.9979** | 3.11% | 55 |
| Hybrid (ST) | **27.20** | 91.47 | 0.9950 | **2.95%** | 52 |
| **Combined (LT)** | 119.22 | **220.86** | **0.9705** | 13.00% | 45 |
| Hybrid (LT) | **117.35** | 247.37 | 0.9635 | **12.74%** | 42 |
| Dynamic Weeks (RF+GBM) | 91.74 | 188.30 | 0.9786 | 10.01% | 15 |

**Key Observations:**
- Combined achieves **35% lower RMSE** than Hybrid on short-term (58.96 vs 91.47), meaning far fewer large prediction errors
- Combined achieves **11% lower RMSE** than Hybrid on long-term (220.86 vs 247.37)
- Combined trades a minor MAE/WMAPE increase (~0.2-0.3%) for substantially better outlier handling
- Dynamic Weeks is a single-horizon model with no short/long distinction; its 10% WMAPE is far worse than either specialized model's short-term performance

### Top Features

**Short-Term Model (Top 10):**

| Rank | Feature | Category |
|---|---|---|
| 1 | diff_1 | Difference |
| 2 | diff_336 | Difference (1 week) |
| 3 | yoy_same_dow_hour_mean | Year-over-Year |
| 4 | lag_1 | Lag (30 min ago) |
| 5 | lag_336 | Lag (7 days ago) |
| 6 | diff_48 | Difference (1 day) |
| 7 | lag_672 | Lag (14 days ago) |
| 8 | inbound_ratio | Channel Mix |
| 9 | callback_ratio | Channel Mix |
| 10 | day_of_month | Temporal |

**Long-Term Model (Top 10):**

| Rank | Feature | Category |
|---|---|---|
| 1 | hist_month_dow_hour_mean | Historical Aggregate |
| 2 | callback_ratio | Channel Mix |
| 3 | hist_week_of_year_mean | Historical Aggregate |
| 4 | hist_month_dow_hour_std | Historical Aggregate |
| 5 | yoy_same_dow_hour_mean | Year-over-Year |
| 6 | day_of_month | Temporal |
| 7 | inbound_ratio | Channel Mix |
| 8 | day_of_year | Temporal |
| 9 | ewm_mean_336 | Long Rolling |
| 10 | rolling_mean_336 | Long Rolling |

### Evaluation Methodology

- **Train/Test Split:** Year-aligned with shared complete months (Jan–Oct 2024 for training, Jan–Oct 2025 for testing) to ensure consistent seasonal distribution
- **Incomplete Month Handling:** If the last month in the test year has fewer than 28 days of data, it is dropped
- **Recency Weighting:** Short-term uses linear weights (0.2 + 0.8 × normalized_index); long-term uses quadratic weights (0.1 + 0.9 × normalized_index²)
- **Primary Metric:** WMAPE (interpretable for staffing); MAE, RMSE, and R² also reported
- **Anomaly Smoothing:** Known outage dates are interpolated before training, preventing corrupted data from affecting model quality

---

## Implementation

### Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **CPU** | 2 cores | 4+ cores |
| **Memory** | 4GB RAM | 8GB RAM |
| **Storage** | 200MB | 500MB |
| **GPU** | Not required | Not required |

### Software Dependencies

```
Python >= 3.9
numpy >= 1.26.0
pandas >= 2.2.0
scikit-learn >= 1.5.0
lightgbm >= 4.0.0
pyarrow >= 14.0.0
```

### Training Configuration

| Parameter | Value |
|---|---|
| Training Data | Jan–Oct 2024 (14,640 intervals) |
| Test Data | Jan–Oct 2025 (14,592 intervals) |
| Feature Scaling | RobustScaler (outlier-resistant) |
| Short-Term Threshold | 7 days |
| Short-Term Recency Weights | Linear: 0.2 + 0.8 × (i / max_i) |
| Long-Term Recency Weights | Quadratic: 0.1 + 0.9 × (i / max_i)² |
| Early Stopping | 50 rounds (both models) |
| Anomaly Smoothing | Linear interpolation for known outage dates |
| Training Time | ~50–60 seconds on Apple M-series |

### Model Hyperparameters

**Short-Term (LGBMRegressor):**

```
n_estimators=800, learning_rate=0.03, num_leaves=127,
max_depth=9, min_child_samples=15, subsample=0.8,
colsample_bytree=0.7, reg_alpha=0.05, reg_lambda=0.5,
early_stopping_rounds=50
```

**Long-Term Ensemble (3× LGBMRegressor):**

```
Model A: n_estimators=1500, lr=0.015, num_leaves=200, max_depth=9,
         subsample=0.8, colsample=0.6, min_child=15, seed=42
Model B: n_estimators=1500, lr=0.015, num_leaves=200, max_depth=9,
         subsample=0.7, colsample=0.5, min_child=15, seed=7
Model C: n_estimators=1500, lr=0.02,  num_leaves=127, max_depth=8,
         subsample=0.85, colsample=0.7, min_child=20, seed=123
All: reg_alpha=1.0, reg_lambda=2.0, early_stopping_rounds=50
```

### Usage

**Training:**

```python
from main_module.workforce.combined_forecaster import CombinedForecaster

forecaster = CombinedForecaster()
forecaster.train("data/raw/dataset_1_call_related.parquet", train_year=2024, test_year=2025)
forecaster.save_model("scripts/combined_forecast_model.pkl")
```

**Inference:**

```python
forecaster = CombinedForecaster()
forecaster.load_model("scripts/combined_forecast_model.pkl")

prediction = forecaster.predict("2025-03-15 14:00:00")

day_forecast = forecaster.predict_day("2025-03-15")
```

---

## Model Data

### Training Data

| Property | Value |
|---|---|
| **Source** | Intuit call center records (QuickBooks / SBSEG) |
| **Primary Dataset** | `dataset_1_call_related.parquet` |
| **Supplementary** | `dataset_3_historical_outcomes.parquet`, `dataset_4_expert_state_interval.parquet` |
| **Time Period** | November 2023 – November 2025 |
| **Total 30-min Intervals** | 34,512 |
| **Train Intervals** | 14,640 (Jan–Oct 2024) |
| **Test Intervals** | 14,592 (Jan–Oct 2025) |

### Data Preprocessing Pipeline

```
Raw Parquet (call-level)
  → Aggregate to 30-min intervals (count + channel ratios)
  → Smooth known anomalies (interpolation)
  → Merge operational metrics from datasets 3 & 4
  → Create base temporal features (incl. holiday/tax features)
  → Compute historical patterns + holiday profiles (training data only)
  → Add lag, rolling, YoY, and operational features
  → Feature scaling (RobustScaler)
```

### Known Anomalies Handled

| Date | Description |
|---|---|
| 2025-08-29 | Full-day data outage; all intervals interpolated |

### Multi-Dataset Integration

| Dataset | Features Extracted |
|---|---|
| dataset_1 (calls) | call_count, inbound_ratio, chat_ratio, callback_ratio |
| dataset_3 (outcomes) | transfer_rate, fcr_rate (first contact resolution), mean_hold |
| dataset_4 (expert state) | active_experts, mean_occupancy, total_available_time |

---

## Integration

### System Architecture

The CombinedForecaster is a drop-in replacement for the HybridForecaster in the three-component pipeline:

```
CombinedForecaster.predict(datetime) → predicted_demand (int)
         │
         ▼
CallCenterEmulator.simulate_interval(supply, demand) → EmulatorMetrics
         │
         ▼
SupplyOptimizer.optimize(demand, constraints) → OptimalSupply (headcount)
```

### Deployment Options

| Method | Description |
|---|---|
| **FastAPI Backend** | `src/main_module/api/main.py` — serves REST API at port 8000; loads pickle on startup |
| **Streamlit Dashboard** | `scripts/dashboard.py` — interactive Python dashboard at port 8501 |
| **React Dashboard** | `src/main_module/visualization/` — TypeScript frontend calling FastAPI at port 3000 |
| **Docker Compose** | `docker-compose.yml` — containerized backend + frontend |
| **CLI Pipeline** | `scripts/run_pipeline.py` — train, forecast, and optimize from command line |

```python
from main_module.workforce.combined_forecaster import CombinedForecaster
forecaster = CombinedForecaster()
```

---

## Ethics and Safety

### Privacy Considerations

- No PII used in features or predictions
- All predictions are aggregated at 30-minute interval level
- Model state does not contain customer information

### Fairness

- Predictions are volume-based, not individual-level
- No demographic features used
- Applies equally across communication channels (inbound, chat, callback)

### Transparency

- Full feature list documented above
- Feature importance computed and reported after each training run
- Training/test split methodology ensures no data leakage
- Year-over-year volume drift explicitly documented
- Known anomalies and their handling are documented

### Environmental Impact

| Metric | Value |
|---|---|
| Training Time | ~53 seconds on Apple M-series CPU |
| Inference Time | < 5ms per interval prediction |
| GPU Required | No |
| Estimated CO₂ | < 0.01 kg per full training |