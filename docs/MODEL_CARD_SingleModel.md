# Model Card: Call Demand Forecaster (Single Model)

## Model Summary

### Overview
| Property | Value |
|----------|-------|
| **Model Name** | CallDemandForecaster |
| **Version** | 1.0 |
| **Type** | Regression (Time Series Forecasting) |
| **Architecture** | Ensemble (VotingRegressor) |
| **File** | `demand_forecasting_model.py` |
| **Saved Model** | `demand_forecast_model.pkl` |

### Description
The Call Demand Forecaster is a machine learning model designed to predict call volume for Intuit's call center operations. It forecasts the number of incoming calls for 30-minute intervals, enabling workforce planning and staffing optimization.

### Architecture
```
┌─────────────────────────────────────────────────────────┐
│                   VotingRegressor                        │
│  (Averages predictions from 3 base models)              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Gradient   │  │   Random    │  │    Ridge    │     │
│  │  Boosting   │  │   Forest    │  │ Regression  │     │
│  │  Regressor  │  │  Regressor  │  │             │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Inputs and Outputs

**Input:**
- Historical call center data (CSV format)
- Target datetime for prediction

**Input Features (67 total):**
| Category | Features | Count |
|----------|----------|-------|
| Temporal | hour, minute, day_of_week, month, quarter, etc. | 16 |
| Cyclical Encoding | hour_sin, hour_cos, dow_sin, dow_cos, etc. | 8 |
| Business Context | is_tax_season, is_tax_deadline, is_holiday, etc. | 12 |
| Lag Features | lag_1, lag_48, lag_336, etc. | 13 |
| Rolling Statistics | rolling_mean_*, rolling_std_*, etc. | 18 |

**Output:**
- Predicted call count (integer ≥ 0) for a 30-minute interval

---

## Model Usage and Limitations

### Intended Usage
- **Primary Use:** Predict call volume for workforce planning
- **Users:** Call center managers, workforce analysts
- **Applications:** 
  - Daily staffing schedules
  - Real-time demand monitoring
  - Capacity planning

### Benefits
- Single unified model for all forecast horizons
- Simpler to deploy and maintain
- Lower computational requirements
- Interpretable feature importance

### Limitations
- **Forecast Horizon:** Accuracy degrades for predictions beyond 1 week
- **Seasonality:** Requires at least 1 year of training data to capture annual patterns
- **Data Dependency:** Relies heavily on recent lag features (lag_48, lag_336)
- **Domain Specific:** Trained specifically for Intuit call center patterns (tax seasonality)
- **Business Hours Only:** Designed for 5am-5pm PT, Monday-Friday

### Out-of-Scope Uses
- Real-time call routing decisions (too slow)
- Predicting individual call outcomes
- Non-call-center demand forecasting without retraining

### Known Biases
- Higher error during tax deadline periods (Apr 10-15) due to extreme volume spikes
- Lower accuracy for holiday periods not well-represented in training data

---

## Evaluation

### Performance Metrics

**Test Set Performance (2024 data, trained on 2023):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 0.78 calls | Average prediction error |
| **RMSE** | 2.02 calls | Penalizes large errors |
| **R²** | 0.9946 | Variance explained |
| **MAPE** | ~24% | Percentage error |

### Cross-Validation Results (5-fold walk-forward)

| Model | CV MAE | CV RMSE | CV R² |
|-------|--------|---------|-------|
| Ridge | 0.409 ± 0.092 | 0.694 | 0.9945 |
| Gradient Boosting | 0.931 ± 0.477 | 2.084 | 0.9662 |
| Random Forest | 1.019 ± 0.486 | 2.260 | 0.9606 |
| **Ensemble** | 0.786 ± 0.352 | 1.679 | 0.9738 |

### Evaluation Methodology
- **Train/Test Split:** Chronological (Train: 2023, Test: 2024)
- **Cross-Validation:** Walk-forward with expanding window
- **Metrics:** MAE (primary), RMSE, R², MAPE

### Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | lag_336 (last week) | 0.391 |
| 2 | lag_same_time_last_week | 0.377 |
| 3 | is_open | 0.051 |
| 4 | is_business_hours | 0.048 |
| 5 | tax_urgency | 0.024 |
| 6 | days_to_tax_deadline | 0.023 |
| 7 | lag_1 | 0.012 |
| 8 | rolling_mean_48 | 0.010 |
| 9 | ewm_std_48 | 0.009 |
| 10 | diff_336 | 0.009 |

---

## Implementation

### Hardware Requirements
- **Training:** Standard CPU (no GPU required)
- **Memory:** 4GB+ RAM recommended
- **Storage:** ~50MB for model file

### Software Dependencies
```
Python >= 3.9
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0 (optional)
lightgbm >= 3.3.0 (optional)
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Training Data | 2023 (17,462 intervals) |
| Test Data | 2024 (17,506 intervals) |
| Feature Scaling | StandardScaler |
| Cross-Validation | 5-fold walk-forward |

### Model Hyperparameters

**Gradient Boosting:**
```python
n_estimators=200, max_depth=6, learning_rate=0.1,
min_samples_split=10, min_samples_leaf=5, subsample=0.8
```

**Random Forest:**
```python
n_estimators=150, max_depth=12, min_samples_split=10,
min_samples_leaf=5
```

**Ridge:**
```python
alpha=1.0
```

---

## Model Data

### Training Data
| Property | Value |
|----------|-------|
| **Source** | Intuit call center records (synthetic) |
| **Time Period** | January 2023 - December 2023 |
| **Total Records** | ~250,000 calls |
| **Intervals** | 17,462 (30-minute intervals) |
| **Products** | TurboTax, QuickBooks |

### Data Preprocessing
1. Aggregate raw calls to 30-minute intervals
2. Fill missing intervals with 0 calls
3. Create temporal and lag features
4. Apply StandardScaler normalization

### Data Distribution
- **Peak Season:** January-April (tax season)
- **Off-Peak:** May-December
- **Business Hours:** 5am-5pm PT, Monday-Friday

---

## Ethics and Safety

### Privacy Considerations
- Model does not use personally identifiable information (PII)
- Predictions are aggregated (no individual customer data)
- Training data is synthetic/anonymized

### Fairness
- Model predicts volume, not individual behavior
- No demographic features used
- Equal treatment across product lines

### Environmental Impact
- Training time: ~2 minutes on standard CPU
- Low computational footprint
- No GPU required

---

## Terms and Links

### Model Card Version
- **Created:** February 2024
- **Last Updated:** February 2024
- **Author:** DSC180A Capstone Team

### Related Resources
- [Technical Documentation](TECHNICAL_DOCUMENTATION.md)
- [Project Documentation](PROJECT_DOCUMENTATION.md)
- [Hybrid Model Card](MODEL_CARD_HybridModel.md)

### Citation
```
@misc{intuit_call_forecaster_single,
  title={Call Demand Forecaster: Single Model},
  author={DSC180A Capstone Team},
  year={2024},
  institution={UC San Diego}
}
```
