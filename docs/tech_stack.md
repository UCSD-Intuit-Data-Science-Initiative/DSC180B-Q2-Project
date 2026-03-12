---
layout: default
title: Tech Stack & Tooling
nav_order: 4
---

# Tech Stack & Tooling

_This page details the technologies, services, and infrastructure used to deliver the demand forecasting and workforce optimization project. The architecture comprises a Python/FastAPI backend and a React frontend, fully containerised for reproducible execution._

## Languages & Frameworks

- **Python 3.10+** – Primary language for data processing, machine learning (LightGBM), and the backend API.
- **Node.js 18+** – JavaScript runtime environment required for building and running the frontend dashboard.
- **FastAPI** – High-performance web framework for building the backend API endpoints and serving the model.
- **React + Vite** – Frontend framework and build tool for the interactive Workforce Management Dashboard.
- **Recharts** – Composable charting library built on React components, used for dashboard visualizations (e.g., call volume forecasts, scheduled agents).

## Core Libraries

- **Data manipulation**: `pandas`, `NumPy` for processing 30-minute interval data and feature engineering.
- **Machine learning**: `LightGBM` (`LGBMRegressor`) for short-term and long-term demand forecasting ensembles.
- **Model Serialization**: `pickle` / `joblib` for saving and loading the trained model bundle (`call_volume_model_bundle.pkl`).
- **Data Formats**: `pyarrow` / `fastparquet` for reading `.parquet` datasets.

## Infrastructure & Services


- **Data Storage**: Local file system tracking for raw and processed datasets (e.g., `data/parquet/`, `data/raw/`). *Note: Large data files are ignored by git to avoid bloating the repository and to comply with data governance.*
- **Containerisation**: **Docker** and **Docker Desktop** are used to isolate the backend and frontend environments, ensuring consistent behavior across local development and deployment.
- **Workflow Orchestration**: Automated extraction and processing are managed via Airflow (as detailed in the Data tab). 

## Dev Tooling

- **Poetry (1.8+)**: For deterministic Python dependency management and packaging.
- **npm**: Node package manager for frontend dependencies (run with `--legacy-peer-deps`).
- **Make**: A `Makefile` is utilized to simplify complex commands into simple targets (e.g., `make install`, `make backend-up`).
- **Pre-commit**: Git hooks configured to ensure consistent code quality, formatting, and linting before code is committed.

## DevOps & CI/CD

- **Containerisation (Docker)**: 
  - The backend runs as a FastAPI service exposed on **port 8000**.
  - The frontend runs as a React development server exposed on **port 3000**.
- **Model Deployment Pipeline**: 
  - The model must be explicitly trained and serialized before the application is spun up. 
  - If `call_volume_model_bundle.pkl` is not present at startup, the API will safely fall back to returning placeholder data.

## Access & Security

- **Data Privacy**: Raw `.parquet` files (e.g., `dataset_1_call_related.parquet`) contain sensitive operational metrics and are explicitly excluded from version control (`.gitignore`).
- **Environment Management**: Secrets and API configurations should be managed via `.env` files and securely injected into the Docker containers at runtime.

## Hardware & Local Requirements

To run this project locally, your development machine must meet the following prerequisites:
- **Python**: v3.10 or higher
- **Node.js**: v18 or higher
- **Package Managers**: Poetry v1.8+
- **Container Runtime**: Docker Desktop (if using the containerised deployment)

## Setup & Execution Runbook

**1. Clone and Install Dependencies**
```bash
git clone <repo> && cd <repo>
make install  # Installs Python deps + configures pre-commit hooks

# Install frontend dependencies
cd src/main_module/visualization && npm install --legacy-peer-deps && cd ../../..