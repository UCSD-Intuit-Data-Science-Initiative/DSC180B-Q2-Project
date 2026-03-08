# Workforce Optimization Dashboard — Launch Guide

Step-by-step instructions to run the Intuit Call Center Dashboard locally.

---

## Prerequisites

- **Git** — to clone the repository
- **Docker Desktop** (recommended) — [Download](https://www.docker.com/products/docker-desktop/)
- **OR** Poetry 1.8+ and Node.js — for local (non-Docker) setup

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/UCSD-Intuit-Data-Science-Initiative/DSC180B-Q2-Project.git
cd DSC180B-Q2-Project
```

---

## Step 2: Choose Your Launch Method

### Option A: Docker (Recommended)

**2a. Start the backend**

```bash
make backend-up
```

Wait for the backend to finish starting (about 10–30 seconds). You should see logs indicating the API is ready.

**2b. Start the frontend**

```bash
make frontend-up
```

**2c. Open the dashboard**

- **Dashboard:** http://localhost:3000
- **API docs:** http://localhost:8000/docs

---

### Option B: Local (No Docker)

Use this if Docker causes issues (e.g., data not loading, proxy problems).

**2a. Install dependencies**

```bash
make install
cd src/main_module/visualization && npm install --legacy-peer-deps
cd ../../..
```

**2b. Start the backend** (Terminal 1)

```bash
make backend-local
```

**2c. Start the frontend** (Terminal 2)

```bash
make frontend-local
```

**2d. Open the dashboard**

- **Dashboard:** http://localhost:3000
- **API docs:** http://localhost:8000/docs

---

## Step 3: Stopping the Application

### Docker

```bash
make backend-down
make frontend-down
```

Or stop both at once:

```bash
docker compose down
```

### Local

Press `Ctrl+C` in each terminal where the backend and frontend are running.

---

## Optional: Full Data Setup

By default, the app runs with placeholder data if the model or datasets are missing. For full functionality:

**1. Ensure data files exist**

Place the required parquet files in `data/parquet/` or `data/raw/`:

- `dataset_1_call_related.parquet` — for demand forecasting
- `dataset_4_expert_state_interval.parquet` — for agent analytics and shift scheduler

**2. Train the model**

```bash
PYTHONPATH=src python scripts/train_model.py
```

This creates `data/models/call_volume_model_bundle.pkl`.

**3. Restart the backend**

If using Docker:

```bash
make backend-down
make backend-up
```

If using local:

Stop the backend (`Ctrl+C`) and run `make backend-local` again.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 3000 already in use | Stop the process using port 3000, or run `docker compose down` to stop containers |
| Port 8000 already in use | Stop the process using port 8000 |
| Data stays loading / proxy issues | Use **Option B (Local)** instead of Docker |
| "Model file not found" | Run `PYTHONPATH=src python scripts/train_model.py` (see Optional section) |
| npm install fails | Use `npm install --legacy-peer-deps` |

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `make backend-up` | Start backend (Docker) |
| `make frontend-up` | Start frontend (Docker) |
| `make backend-down` | Stop backend (Docker) |
| `make frontend-down` | Stop frontend (Docker) |
| `make backend-local` | Start backend locally |
| `make frontend-local` | Start frontend locally |
| `make help` | List all available commands |
