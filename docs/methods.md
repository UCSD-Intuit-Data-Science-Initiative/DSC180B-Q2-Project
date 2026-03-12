---
layout: default
title: Methods
nav_order: 2
---

# Modeling Strategy

_This page outlines the methodologies, mathematical frameworks, and key decisions driving our two-stage pipeline: Demand Forecasting and Workforce Optimization._

## Problem Framing

- **Prediction target**: Incoming call volumes aggregated at 30-minute intervals ($C_t$).
- **Business KPIs**: Operational labor costs and customer wait times. The models connect volume predictions to optimal interval-level agent staffing ($N_t$), minimizing minimum headcount as a proxy for labor cost.
- **Constraints**: Staffing optimization is strictly bound by three business thresholds:
  - Service Level (SLA) $\ge$ **0.80** (80% of calls answered within **60 seconds**)
  - Average Wait Time $\le$ **60 seconds**
  - Agent Occupancy $\le$ **0.85** (85%)

## Baseline Model

- **Forecasting Baseline**: We utilize a one-step naïve forecast as our standard baseline to compute the Mean Absolute Scaled Error (MASE). This ensures the model fundamentally outperforms a simple "carry-forward" assumption of the prior interval's volume.
- **Queueing Baseline**: The baseline probability that an arriving call must wait relies on the classic Erlang-C formula:
$$P_{\text{wait}}(N_t, a_t) = \frac{\frac{a_t^{N_t}}{N_t!}\left(\frac{N_t}{N_t-a_t}\right)}{\sum_{k=0}^{N_t-1}\frac{a_t^k}{k!} + \frac{a_t^{N_t}}{N_t!}\left(\frac{N_t}{N_t-a_t}\right)}$$
_(Where $a_t$ is the offered load. Factorial terms are evaluated in log space to avoid overflow.)_

## Feature Engineering

Our forecasting pipeline engineers a comprehensive feature space from raw 30-minute interval data, shifting focus depending on the prediction horizon:

- **Temporal & Cyclical Indicators**: Standard calendar components (hour, day, month) and cyclical transformations (sine/cosine) for smooth period transitions.
- **Tax Seasonality & Holidays**: Binary flags for standard U.S. holidays and custom tax-urgency metrics (e.g., `days_to_tax_deadline` countdown, `is_post_tax_drop` flag) to capture massive ramp-ups before April 15th.
- **Autoregressive Lags (Short-Term)**: Direct volume lags (previous interval, same time yesterday/last week), rolling means, standard deviations, and exponential moving averages to capture intraday momentum.
- **Historical Baselines & Year-Over-Year (Long-Term)**: Historical averages/medians for specific time cross-sections and YoY data to establish baseline expectations without relying on immediate recent momentum.
- **Platform Operational States**: Channel distribution ratios (inbound vs. chat/callback) and expert efficiency indicators (First Call Resolution, mean hold times, occupancy).



## Experiment Tracking

_Forecasting is dynamically routed based on the target date horizon. Models are trained on a 320-day window (Jan 1, 2024 – Nov 15, 2024)._

| Experiment ID | Description | Model/Params | CV/Validation Scheme | Metrics | Notes |
| --- | --- | --- | --- | --- | --- |
| EXP-001 | Baseline Naïve Forecast | One-step naïve | Sequential (Jan-Nov 2025) | MASE = 1.0 | Used for MASE denominator |
| EXP-002 | Short-Term Horizon (< 7 days) | LightGBM Regressor (`n_estimators=800`, `learning_rate=0.03`, `max_depth=9`, `num_leaves=127`) | Temporal sample weighting (linear prioritization of recent data) | MAE, WMAPE | Applies L1/L2 regularization (`reg_alpha=0.05`, `reg_lambda=0.5`) to prevent overfitting on noise. |
| EXP-003 | Long-Term Horizon (7+ days) | LightGBM Ensemble (3 models) | Temporal sample weighting (quadratic curve) | MAE, WMAPE | Averages 3 base learners. Parameters span `n_estimators=1500`, `max_depth=8-9`. Strict regularization applied. |

## Evaluation Strategy

- **Validation Approach**: Sequential time-series testing. Training data is strictly misaligned from the test period to prevent leakage, utilizing an aligned 319-day test period (January 1, 2025 – November 15, 2025) to ensure YoY alignment of tax patterns.
- **Performance Metrics**: 
  - **Mean Absolute Error (MAE)**: Average absolute deviation in calls per day.
  - **Mean Absolute Scaled Error (MASE)**: Normalizes against the naïve baseline.
  $$\text{MASE} = \frac{\text{MAE}_{\text{model}}}{\frac{1}{n-1}\sum_{t=2}^{n}|y_t - y_{t-1}|}$$
  - **Weighted Mean Absolute Percentage Error (WMAPE)**: Expresses error as a percentage of total actual volume.

## Advanced Techniques: Workforce Optimization & Queueing



To translate forecasted volumes into staffing requirements, we treat interval-level staffing as a discrete optimization problem. 

### Abandonment-Aware Performance Emulator
Instead of slow discrete-event simulation, the optimizer queries a deterministic analytical emulator. It modifies the standard Erlang-C wait probability with an Erlang-A-inspired adjustment for caller abandonment, assuming an average patience ($\eta$) of **180 seconds**:
$$P_{\text{abandon}}(N_t) \approx P_{\text{wait}}(N_t, a_t) \cdot \frac{\theta_t a_t}{N_t(1-\rho_t) + \theta_t a_t}$$
_(Where $\theta_t = \frac{\eta}{\mu_t}$ and $\rho_t = \frac{a_t}{N_t}$)_

Wait time and SLA metrics are subsequently derived from these abandonment-adjusted probabilities:
$$\widehat{W}_t(N_t) \approx \frac{P_{\text{wait}}(N_t, a_t)}{N_t \mu_t (1-\rho_t) + \eta}$$
$$\widehat{SL}_t(N_t) \approx 1 - P_{\text{wait}}(N_t, a_t)\exp\left[-\left(N_t\mu_t(1-\rho_t)+\eta\right)T\right] - P_{\text{abandon}}(N_t)$$

### Optimization Algorithm
Since queueing performance monotonically improves with staffing, we locate the minimum feasible headcount ($N_t^\star$) using **binary search** instead of a linear scan. 
1. We compute an occupancy-aware lower bound: $N_{\min} = \max\left(1,\left\lceil \frac{a_t}{\tau_\rho} \right\rceil + 1\right)$.
2. We evaluate midpoints between $N_{\min}$ and a safety upper bound $N_{\max}$.
3. The minimum feasible integer satisfying SLA, Wait Time, and Occupancy thresholds is selected.

$$N_t^\star = \min \left\{ N \in \mathbb{Z}_{\ge 0} : \widehat{SL}_t(N) \ge \tau_{SL}, \ \widehat{W}_t(N) \le \tau_W, \ \widehat{\rho}_t(N) \le \tau_\rho \right\}$$

## Deployment Considerations

- **Pipeline Integration**: The workforce optimization module directly connects the LightGBM forecaster to downstream operational planning. For every interval $t$, the system predicts $C_t$, retrieves historical Average Handle Time ($H_t$), computes $N_t^\star$, and outputs recommended headcounts alongside emulator metrics.
- **Handling Edge Cases**: The optimization module contains logic guards for zero-demand intervals (returns $N_t^\star = 0$) and overloaded intervals (bounds overflow rather than evaluating unstable queueing expressions). 
- **Downstream Usage**: Interval-level outputs act as operational benchmarks for downstream shift scheduling modules, dictating the construction of actual agent shifts based on historical availability patterns.