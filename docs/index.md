---
layout: default
title: Introduction
nav_order: 1
---

# Workforce Optimization for Intuit's Virtual Expert Platform

**Authors:** 

Jackie Wang (`jaw039@ucsd.edu`)

Sophia Fang (`sofang@ucsd.edu`)

Sarah He (`yih092@ucsd.edu`)

Hao Zhang (`haz105@ucsd.edu`)

**Mentors:** Akash Shah, Victor Calderon, Joe Cessna, Ken Yocum, Lisa Li

---

### Abstract

Intuit's Virtual Expert Platform (VEP) requires efficient workforce planning to balance operational costs with high-quality customer support, particularly during highly variable, high-volume tax seasons. To address the inefficiencies of reactive scheduling, we developed an end-to-end workforce optimization pipeline that bridges machine learning with operational queueing theory. First, a dual-horizon demand forecasting architecture predicts incoming call volumes at 30-minute intervals. A short-term LightGBM model tracks intraday momentum, while a long-term LightGBM ensemble maps macro-seasonal tax trends and historical baselines. These volume predictions are then processed by an abandonment-aware performance emulator, an adaptation of the Erlang-A model, to calculate the minimum agent headcount required to satisfy strict Service Level Agreements, maximum wait times, and maximum occupancy thresholds. Finally, these backend computations are integrated into an interactive frontend dashboard that enables workforce managers to monitor real-time productivity, simulate "what-if" operational scenarios, and programmatically schedule shifts. Ultimately, this prototype transforms workforce management from a reactive guessing game into a proactive, data-driven process, ensuring Intuit can minimize unnecessary labor overhead while reliably meeting customer needs.

**Code:** [GitHub Repository](https://github.com/UCSD-Intuit-Data-Science-Initiative/DSC180B-Q2-Project)

---

## Introduction

Intuit, a leading financial software company, offers products like TurboTax and QuickBooks, and connects users with live experts through its Virtual Expert Platform (VEP). This platform’s operational efficiency relies heavily on accurate demand forecasting and precise supply planning. By effectively matching expert availability with user needs, Intuit ensures timely support and maintains cost-effectiveness. Forecasting models in workforce management set the stage for everything else. Accuracy matters because small errors in forecasts ripple downstream, leading to overstaffing (higher costs) or understaffing (longer wait times and lower customer satisfaction). On the supply planning side, the challenge is about optimization under constraints, balancing multiple conflicting requirements such as skill mixes and labor regulations.

### Literature Review & Discussion of Prior Work

* **Call Volume Forecasting:** Prior studies suggest that while statistical time series models provide a strong baseline, incorporating operational factors such as marketing events can significantly improve forecast accuracy.
* **Workforce Supply Optimization:** Research highlights that call centers face complexities beyond typical optimization, such as time-varying demand and human factors like agent burnout.