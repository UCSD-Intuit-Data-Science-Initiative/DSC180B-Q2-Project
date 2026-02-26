# ---- Dev stage (used by docker-compose / make backend-up) ----
FROM python:3.11-slim AS dev

WORKDIR /app

# Copy dependency list and install
# Docker caches this layer — if requirements.txt hasn't changed,
# pip install is skipped on rebuild (makes rebuilds faster)
COPY src/main_module/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tell Python where to find our main_module package
# Without this, `from main_module.workforce import ...` would fail
ENV PYTHONPATH=/app/src

EXPOSE 8000

# --reload  = restarts the server automatically when you edit a .py file (hot reload)
# --host 0.0.0.0 = required for Docker so the port is reachable from your laptop
CMD ["python", "-m", "uvicorn", "main_module.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# ---- Production stage ----
FROM python:3.11-slim AS production

WORKDIR /app

COPY src/main_module/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app/src

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main_module.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
