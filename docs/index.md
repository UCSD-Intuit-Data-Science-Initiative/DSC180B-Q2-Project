# Workforce Optimization for Intuit's Virtual Expert Platform

**Authors:** Jackie Wang (`jaw039@ucsd.edu`)

Sophia Fang (`sofang@ucsd.edu`)

Sarah He (`yih092@ucsd.edu`)

Hao Zhang (`haz105@ucsd.edu`)

**Mentors:** Akash Shah, Victor Calderon, Joe Cessna, Ken Yocum, Lisa Li

---

### Abstract

Intuit's Virtual Expert Platform (VEP) requires efficient workforce planning to balance operational costs with high-quality customer support, particularly during highly variable, high-volume tax seasons. To address the inefficiencies of reactive scheduling, we developed an end-to-end workforce optimization pipeline that bridges machine learning with operational queueing theory. First, a dual-horizon demand forecasting architecture predicts incoming call volumes at 30-minute intervals. A short-term LightGBM model tracks intraday momentum, while a long-term LightGBM ensemble maps macro-seasonal tax trends and historical baselines. These volume predictions are then processed by an abandonment-aware performance emulator, an adaptation of the Erlang-A model, to calculate the minimum agent headcount required to satisfy strict Service Level Agreements, maximum wait times, and maximum occupancy thresholds. Finally, these backend computations are integrated into an interactive frontend dashboard that enables workforce managers to monitor real-time productivity, simulate "what-if" operational scenarios, and programmatically schedule shifts. Ultimately, this prototype transforms workforce management from a reactive guessing game into a proactive, data-driven process, ensuring Intuit can minimize unnecessary labor overhead while reliably meeting customer needs.

**Code:** [GitHub Repository](https://github.com/UCSD-Intuit-Data-Science-Initiative/DSC180B-Q2-Project)

---

## 1. Introduction

Intuit, a leading financial software company, offers products like TurboTax and QuickBooks, and connects users with live experts through its Virtual Expert Platform (VEP). This platform’s operational efficiency relies heavily on accurate demand forecasting and precise supply planning. By effectively matching expert availability with user needs, Intuit ensures timely support and maintains cost-effectiveness. Forecasting models in workforce management set the stage for everything else. Accuracy matters because small errors in forecasts ripple downstream, leading to overstaffing (higher costs) or understaffing (longer wait times and lower customer satisfaction). On the supply planning side, the challenge is about optimization under constraints, balancing multiple conflicting requirements such as skill mixes and labor regulations.

### 1.1 Literature Review & Discussion of Prior Work

* **Call Volume Forecasting:** Prior studies suggest that while statistical time series models provide a strong baseline, incorporating operational factors such as marketing events can significantly improve forecast accuracy.
* **Workforce Supply Optimization:** Research highlights that call centers face complexities beyond typical optimization, such as time-varying demand and human factors like agent burnout.

### 1.2 Description of Relevant Data

Our datasets contain comprehensive daily call center operational metrics covering two years from November 27, 2023, to November 15, 2025.

*(Tables 1 through 4 would be inserted here)*

---

## 2. Methods

### 2.1 Demand Forecasting

Our pipeline predicts incoming call volumes at 30-minute intervals, evaluating a 319-day test period in 2025 against a 320-day training window from 2024 to ensure year-over-year seasonal alignment.

#### 2.1.1 Evaluation Framework

We use three primary metrics:

* **Mean Absolute Error (MAE)**
* **Mean Absolute Scaled Error (MASE)**
* **Weighted Mean Absolute Percentage Error (WMAPE)**

#### 2.1.2 Feature Engineering

* **Temporal & Cyclical Indicators:** Standard calendar components and sine/cosine transformations.
* **Tax Seasonality & Holidays:** Flags for holidays and custom metrics like `days_to_tax_deadline`.
* **Autoregressive Lags (Short-Term):** Immediate historical behavior (e.g., volume from the previous interval).
* **Historical Baselines (Long-Term):** Average and median volumes for specific time cross-sections and YoY data.
* **Platform Operational States:** Metrics like channel distribution and expert efficiency (First Call Resolution).

#### 2.1.3 Machine Learning Models

* **Short-Term LightGBM:** For horizons < 7 days, relying on recent lags and intraday momentum.
* **Long-Term LightGBM Ensemble:** For horizons ≥ 7 days, using an ensemble of three models to capture macro-seasonal patterns.

### 2.2 Workforce Optimization

#### 2.2.1 Interval-Level Problem Formulation

We solve for the minimum headcount $N_t^\star$ for each 30-minute interval subject to:

* **Service Level (SL)** $\ge 80\%$
* **Average Wait Time (W)** $\le 60s$
* **Agent Occupancy ($\rho$)** $\le 85\%$

#### 2.2.2 Abandonment-Aware Performance Emulator

The staffing optimizer queries a deterministic analytical emulator. It combines Erlang-C waiting probability with an Erlang-A-inspired abandonment adjustment, assuming an average patience time of 180 seconds.

#### 2.2.3 Optimization Algorithm

Because performance improves monotonically with staffing, we use **Binary Search** to find the minimum feasible headcount for each interval efficiently.

---

## 3. Results

### 3.1 Forecasting Model Performance

* **Short-Term Horizon:** WMAPE of **3.11%** and $R^2$ of **0.9979**. High accuracy driven by autoregressive signals.
* **Long-Term Horizon:** WMAPE of **13.12%** and $R^2$ of **0.9706**. Driven primarily by historical averages and macro-seasonality.

### 3.2 Workforce Optimization Results

Evaluated during the peak tax week (April 14–18, 2025), the optimizer achieved **100% feasibility** across all intervals.

* **Mean SLA:** 99.85%
* **Mean Wait Time:** 0.07 seconds
* **Occupancy Range:** 83.47% – 84.97%

### 3.3 Workforce Management Dashboard

We developed an interactive dashboard to translate these backend computations into a practical tool for managers.

* **Main Dashboard:** High-level overview of performance and forecasts.
* **Performance Simulator:** Allows "what-if" analysis by adjusting SLA or occupancy targets.
* **Shift Scheduler:** Timeline-based interface for assigning agents to shifts based on required staffing levels.

*(See Appendix for Interface Screenshots)*

---

## 4. Discussion & Conclusion

### 4.1 Limitations

The system assumes stationary arrival rates within 30-minute blocks and uses a uniform Average Handle Time (AHT). Future work could model handle-time variance and account for unpredictable platform outages.

### 4.2 Conclusion

This prototype transforms workforce management from reactive guessing into proactive, data-driven planning. By integrating dual-horizon forecasting with an abandonment-aware optimizer, Intuit can reduce labor overhead while ensuring customers receive timely support during critical high-volume periods.

---

## 5. Contributions

* **Hao Zhang:** End-to-end data pipelines, forecasting fine-tuning, and shift scheduler logic.
* **Jackie Wang:** Project architecture, backend integration, and emulator logic.
* **Sarah He:** Demand forecasting models, frontend integration for the Simulator, and report drafting.
* **Sophia Fang:** Design and development of the full interactive dashboard suite (UI/UX and implementation).

---

## Appendix: Dashboard Screenshots

**Figure 1: Main Dashboard**

*Weekly performance metrics, demand forecast, and agent productivity summary.*

**Figure 2: Weekly Forecast**

**Figure 3: Performance Simulator**

*Managers can adjust SLA, wait time, and occupancy targets.*

**Figure 4: Shift Scheduler Timeline**

*Interactive timeline showing agent shifts and required staffing levels.*
