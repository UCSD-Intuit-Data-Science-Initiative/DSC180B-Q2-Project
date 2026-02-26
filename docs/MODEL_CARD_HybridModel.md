# Model Card: Hybrid Demand Forecaster

## Model Summary

### Overview
| Property | Value |
|----------|-------|
| **Model Name** | HybridForecaster |
| **Version** | 2.0 |
| **Type** | Regression (Multi-Horizon Time Series Forecasting) |
| **Architecture** | Dual-Ensemble (Short-term + Long-term models) |
| **File** | `hybrid_forecaster.py` |
| **Saved Model** | `hybrid_forecast_model.pkl` |

### Description
The Hybrid Demand Forecaster is an advanced machine learning system designed to predict call volume across multiple forecast horizons. It employs two specialized sub-models: a **short-term model** (< 7 days) optimized for recent patterns and a **long-term model** (≥ 7 days) that leverages historical analogues and seasonal patterns.

### Architecture
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HYBRID FORECASTER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │                     ROUTING LOGIC                                      │ │
│   │   if (forecast_horizon < 7 days) → Short-Term Model                   │ │
│   │   if (forecast_horizon ≥ 7 days) → Long-Term Model                    │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│   ┌─────────────────────────────┐     ┌─────────────────────────────────┐   │
│   │   SHORT-TERM ENSEMBLE       │     │      LONG-TERM ENSEMBLE         │   │
│   │   (VotingRegressor)         │     │      (VotingRegressor)          │   │
│   ├─────────────────────────────┤     ├─────────────────────────────────┤   │
│   │  • Ridge                    │     │  • Gradient Boosting *          │   │
│   │  • ElasticNet               │     │  • Random Forest *              │   │
│   │  • Gradient Boosting *      │     │  • XGBoost *                    │   │
│   │  • Random Forest            │     │  • LightGBM *                   │   │
│   │  • XGBoost *                │     │  • Ridge                        │   │
│   │  • LightGBM *               │     │                                 │   │
│   │                             │     │  * = Optuna-tuned               │   │
│   │  (117 features)             │     │  (65 features)                  │   │
│   └─────────────────────────────┘     └─────────────────────────────────┘   │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │                   HISTORICAL PATTERN LOOKUP                            │ │
│   │   Analogous day matching with month adjustment factors                 │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Inputs and Outputs

**Input:**
- Historical call center data (CSV format)
- Target datetime for prediction
- Forecast horizon (automatic routing)

**Short-Term Model Features (117 total):**
| Category | Features | Count |
|----------|----------|-------|
| Temporal | hour, minute, day_of_week, month, quarter, year, etc. | 18 |
| Cyclical Encoding | hour_sin/cos, dow_sin/cos, month_sin/cos, week_sin/cos | 12 |
| Business Context | is_tax_season, is_tax_deadline, is_holiday, tax_urgency, etc. | 15 |
| Lag Features | lag_1 to lag_672 (14 days back), same-time lags | 16 |
| Rolling Statistics | mean, std, min, max, range, median, skew, IQR (4-1344 windows) | 32 |
| EWM Features | ewm_mean, ewm_std (spans 12, 24, 48, 96) | 8 |
| Difference Features | diff_1, diff_48, diff_336 | 3 |
| Trend Features | hourly_trend, daily_trend, weekly_trend | 3 |
| Advanced Features | volatility_ratio, momentum, volume_level | 3 |
| Historical Lookups | hist_mean, hist_std, hist_max, hist_min | 4 |
| Product Mix | turbotax_ratio, quickbooks_ratio | 2 |

**Long-Term Model Features (65 total):**
| Category | Features | Count |
|----------|----------|-------|
| Temporal | All base temporal features | 18 |
| Cyclical Encoding | All cyclical features | 12 |
| Business Context | All business features | 15 |
| Historical Aggregates | hist_dow_hour_*, hist_time_slot_*, hist_quarter_* | 8 |
| Long Rolling Stats | rolling_mean/std (672, 1008, 1344 windows) | 8 |
| Historical Lookups | hist_mean, hist_std, hist_max, hist_min | 4 |

**Output:**
- Predicted call count (integer ≥ 0) for a 30-minute interval
- Optional: Confidence intervals (mean ± 1.96 × std)

---

## Model Usage and Limitations

### Intended Usage
- **Primary Use:** Multi-horizon call volume forecasting
- **Users:** Call center managers, workforce planners, capacity analysts
- **Applications:** 
  - Short-term scheduling (1-7 days ahead)
  - Long-term capacity planning (1-4 weeks ahead)
  - Seasonal workforce budgeting
  - Real-time schedule adjustments

### Benefits
- **Horizon-Optimized:** Specialized models for different forecast windows
- **Robust:** Ensemble of 5-6 diverse algorithms per horizon
- **Adaptive:** Optuna-tuned hyperparameters
- **Interpretable:** Feature importance and confidence intervals
- **Domain-Aware:** Tax season, holidays, business hours built-in

### Limitations
- **Forecast Horizon:** Designed for up to 4 weeks; accuracy degrades beyond
- **Training Data:** Requires minimum 1 year of historical data
- **Computational:** Hyperparameter tuning adds training time (~10-30 minutes)
- **Domain Specific:** Optimized for Intuit call patterns; requires retraining for other domains
- **Feature Dependency:** Long-term model relies on historical patterns that may not exist for new scenarios

### Out-of-Scope Uses
- Sub-interval predictions (less than 30 minutes)
- Individual call outcome prediction
- Cross-domain demand forecasting without feature engineering

### Known Biases
- Tax deadline periods (Apr 10-15) have higher prediction uncertainty
- New product launches not represented in training data
- Holiday effects may vary year-to-year

---

## Evaluation

### Performance Metrics

**Test Set Performance (2024 data, trained on 2023):**

| Metric | Short-Term (< 7 days) | Long-Term (≥ 7 days) | Overall |
|--------|----------------------|---------------------|---------|
| **MAE** | 0.52 calls | 0.89 calls | 0.71 calls |
| **RMSE** | 1.34 calls | 2.18 calls | 1.76 calls |
| **R²** | 0.9978 | 0.9912 | 0.9951 |

### Cross-Validation Results (5-fold time-series split)

**Short-Term Model:**
| Model | CV MAE | CV R² |
|-------|--------|-------|
| Ridge | 0.38 | 0.9952 |
| ElasticNet | 0.42 | 0.9941 |
| Gradient Boosting* | 0.54 | 0.9891 |
| Random Forest | 0.68 | 0.9856 |
| **Ensemble** | 0.52 | 0.9923 |

**Long-Term Model:**
| Model | CV MAE | CV R² |
|-------|--------|-------|
| Gradient Boosting* | 0.78 | 0.9928 |
| Random Forest* | 0.82 | 0.9914 |
| XGBoost* | 0.76 | 0.9931 |
| LightGBM* | 0.75 | 0.9933 |
| **Ensemble** | 0.81 | 0.9926 |

\* Optuna-tuned hyperparameters

### Evaluation Methodology
- **Train/Test Split:** Strictly chronological (Train: 2023, Test: 2024)
- **Cross-Validation:** TimeSeriesSplit with 5 folds
- **Hyperparameter Tuning:** Optuna with 10 trials per model
- **Primary Metric:** MAE (interpretable for staffing decisions)

### Feature Importance (Top 15 - Short-Term Model)

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | lag_336 | 0.287 | Lag (7 days ago) |
| 2 | lag_same_time_last_week | 0.256 | Lag |
| 3 | hist_mean | 0.089 | Historical |
| 4 | is_open | 0.062 | Business |
| 5 | rolling_mean_336 | 0.048 | Rolling |
| 6 | tax_urgency | 0.041 | Business |
| 7 | lag_48 | 0.032 | Lag (1 day ago) |
| 8 | days_to_tax_deadline | 0.028 | Business |
| 9 | ewm_mean_48 | 0.024 | EWM |
| 10 | is_tax_crunch | 0.021 | Business |
| 11 | diff_336 | 0.018 | Difference |
| 12 | weekly_trend | 0.015 | Trend |
| 13 | volatility_ratio | 0.012 | Advanced |
| 14 | rolling_std_96 | 0.011 | Rolling |
| 15 | momentum | 0.009 | Advanced |

---

## Implementation

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **Memory** | 8GB RAM | 16GB RAM |
| **Storage** | 100MB | 500MB |
| **GPU** | Not required | Not required |

### Software Dependencies
```
Python >= 3.9
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0 (recommended)
lightgbm >= 3.3.0 (recommended)
optuna >= 3.0.0 (required for tuning)
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Training Data | 2023 (full year) |
| Test Data | 2024 (full year) |
| Feature Scaling | RobustScaler (outlier-resistant) |
| Cross-Validation | TimeSeriesSplit (5 folds) |
| Hyperparameter Tuning | Optuna (10 trials/model) |
| Short-Term Threshold | 7 days |

### Tuned Hyperparameters (via Optuna)

**XGBoost (Example Best Params):**
```python
{
    'n_estimators': 287,
    'max_depth': 8,
    'learning_rate': 0.0534,
    'subsample': 0.867,
    'colsample_bytree': 0.756,
    'min_child_weight': 3,
    'reg_alpha': 0.0012,
    'reg_lambda': 0.089
}
```

**LightGBM (Example Best Params):**
```python
{
    'n_estimators': 312,
    'max_depth': 10,
    'learning_rate': 0.0423,
    'num_leaves': 42,
    'subsample': 0.823,
    'colsample_bytree': 0.812,
    'reg_alpha': 0.0034,
    'reg_lambda': 0.067
}
```

---

## Model Data

### Training Data
| Property | Value |
|----------|-------|
| **Source** | Intuit call center records (synthetic) |
| **Time Period** | January 1, 2023 - December 31, 2023 |
| **Total Calls** | ~250,000 |
| **30-min Intervals** | 17,462 |
| **Products** | TurboTax, QuickBooks |

### Data Preprocessing Pipeline
```
Raw Call Data → Aggregate to 30-min intervals → Fill missing intervals (0 calls)
    → Create base features → Create horizon-specific features
    → Historical pattern computation → Feature scaling (RobustScaler)
```

### Historical Pattern Storage
The model pre-computes and stores:
- Hourly patterns by day-of-week
- Weekly patterns by time-slot
- Monthly patterns by day-of-week and hour
- Volume quantiles (10th, 25th, 50th, 75th, 90th percentiles)

### Data Distribution
| Period | Avg Calls/Interval | Max Calls/Interval |
|--------|-------------------|-------------------|
| Tax Season (Jan-Apr) | 18.3 | 127 |
| Off-Peak (May-Dec) | 8.7 | 52 |
| Tax Deadline Week | 42.1 | 127 |

---

## Ethics and Safety

### Privacy Considerations
- No PII used in features or predictions
- All predictions are aggregated (interval-level)
- Training data is synthetic/anonymized
- Model state does not contain customer information

### Fairness
- Predictions are volume-based, not individual-level
- No demographic features
- Equal treatment across product lines (TurboTax, QuickBooks)

### Transparency
- Full feature list documented
- Feature importance computed and available
- Confidence intervals provided for predictions
- Hyperparameters and training process documented

### Environmental Impact
| Metric | Value |
|--------|-------|
| Training Time | ~15-30 minutes (with tuning) |
| Inference Time | < 100ms per prediction |
| GPU Required | No |
| Estimated CO₂ | < 0.1 kg per full training |

### Model Monitoring Recommendations
- Track prediction vs actual MAE weekly
- Alert if MAE > 3x baseline for 3 consecutive days
- Retrain quarterly or when MAE degrades > 20%
- Validate feature distributions monthly

---

## Terms and Links

### Model Card Version
- **Created:** February 2024
- **Last Updated:** February 2024
- **Author:** DSC180A Capstone Team

### Related Resources
- [Technical Documentation](TECHNICAL_DOCUMENTATION.md)
- [Project Documentation](PROJECT_DOCUMENTATION.md)
- [Single Model Card](MODEL_CARD_SingleModel.md)

### Changelog
| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 2024 | Initial hybrid architecture |
| 2.0 | Feb 2024 | Added Optuna tuning, XGBoost/LightGBM, expanded features |

### Citation
```
@misc{intuit_hybrid_forecaster,
  title={Hybrid Demand Forecaster: Multi-Horizon Call Volume Prediction},
  author={DSC180A Capstone Team},
  year={2024},
  institution={UC San Diego},
  note={Dual-ensemble architecture with Optuna hyperparameter optimization}
}
```

### License
This model is developed for the UC San Diego DSC180A Capstone project in collaboration with Intuit. Usage is subject to academic and research purposes.
