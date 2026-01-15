---
layout: default
title: Tech Stack & Tooling
nav_order: 5
---

# Tech Stack & Tooling

_Detail the technologies, services, and infrastructure used to deliver this project. Keep it current as architecture evolves._

## Languages & Frameworks

- **Python 3.11+** – primary language for data processing and modelling.
- **(Optional)** R, SQL, Scala – document if applicable.
- **Visualization** – Matplotlib, Seaborn, Plotly (list whichever you adopt).

## Core Libraries

- Data manipulation: pandas, NumPy.
- Machine learning: scikit-learn, XGBoost, LightGBM, TensorFlow/PyTorch (specify).
- Feature engineering: category encoders, pipeline utilities.
- Evaluation: scikit-learn metrics, statsmodels.

## Infrastructure & Services

- Data storage: S3 buckets, Azure Blob, GCS, PostgreSQL, etc.
- Compute: local dev machines, cloud instances, Databricks, Spark clusters.
- Workflow orchestration: Airflow, Prefect, Dagster.
- Experiment tracking: MLflow, Weights & Biases.
- Monitoring: custom dashboards, logging sinks.

## Dev Tooling

- **Poetry** for dependency management and packaging.
- **Ruff + mypy** for linting, formatting, typing.
- **Tox** for environment orchestration.
- **Pytest + pytest-cov** for testing/coverage.
- **Loguru** for logging.
- **Pre-commit** hooks for consistent code quality.

## DevOps & CI/CD

- Continuous integration (e.g., GitHub Actions) – describe jobs/pipelines if configured.
- Containerisation (Docker) – note base images, build process.
- Deployment targets (API service, batch jobs, dashboards).

## Access & Security

- Secrets management (environment variables, vaults, AWS Secrets Manager).
- IAM roles/permissions needed for cloud resources.
- Audit logging and compliance requirements.

## Hardware Requirements

- Local development specs (RAM, CPU, GPU if required).
- Production environment specs.

## References

- Links to architecture diagrams, ADRs (Architecture Decision Records), or runbooks.
- Vendor documentation or internal wiki pages.
