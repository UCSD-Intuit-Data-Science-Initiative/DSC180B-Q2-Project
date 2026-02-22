---
layout: default
title: Data & Experimentation
nav_order: 1
---

# Data & Experimentation

_Use this page to describe the datasets, their provenance, data quality considerations, and how exploratory analysis is conducted._

## Data Inventory

| Dataset | Source | Location | Notes |
| --- | --- | --- | --- |
| Example Raw Dataset | Internal CRM export | `data/raw/example.csv` | Daily extract; contains PII (mask or anonymize) |
| Processed Features | Derived via `clean_dataframe` | `data/processed/features.parquet` | Updated after each ETL run |
| External Reference | Public API | `data/external/reference.json` | Refresh monthly |

Keep this table current as new datasets are added or existing ones change.

## Generation & Extraction

- **Acquisition scripts**: document where code lives (e.g., `scripts/`, external ETL).
- **Refresh cadence**: daily/weekly/monthly? Automate via cron, Airflow, etc.
- **Access requirements**: credentials, VPN, API tokens. Store secrets securely (never commit).

### Placeholder text

```
Describe how raw data is pulled. Include sample queries, API endpoints,
and expected schemas. Note any governance requirements or limits.
```

## Data Quality & Profiling

- Outline checks for missing values, duplicates, schema drift.
- Record profiling summaries (e.g., pandas-profiling, Great Expectations).
- Log anomalies and remediation steps.

## Exploratory Analysis

- Notebook convention (e.g., `01.0-abc-eda.ipynb`).
- Where to store outputs (plots, tables) during exploration.
- How insights feed back into feature engineering or modelling.

## Transformation Pipeline

- Reference helper functions/modules (`load_csv`, `clean_dataframe`, `select_columns`).
- Describe feature engineering steps, transformations, encodings, scaling.
- Maintain versioned transformation logic (config files, code branches).

## Data Governance

- PII handling (masking, hashing, removal).
- Compliance requirements (GDPR, CCPA, internal policies).
- Approval process for sharing data externally.
