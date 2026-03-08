.PHONY: help install lint format test docs-up docs-down docs-logs docs-check docs-build docs-ps clean precommit backend-up backend-down backend-logs backend-open

POETRY ?= poetry

PODMAN_BIN := $(shell command -v podman 2>/dev/null)
PODMAN_COMPOSE_BIN := $(shell command -v podman-compose 2>/dev/null)
DOCKER_BIN := $(shell command -v docker 2>/dev/null)

ifeq ($(strip $(PODMAN_BIN)),)
    CONTAINER_ENGINE := docker
    COMPOSE_CMD := docker compose
else
    CONTAINER_ENGINE := podman
    ifneq ($(strip $(PODMAN_COMPOSE_BIN)),)
        COMPOSE_CMD := podman-compose
    else
        COMPOSE_CMD := podman compose
    endif
endif

DOCS_COMPOSE_CMD := $(COMPOSE_CMD) -f docker-compose.docs.yml

.DEFAULT_GOAL := help


##@Setup
install: ## Install project and dev dependencies
	$(POETRY) install
	$(POETRY) run pre-commit install

##@Quality
lint: ## Run static analysis with Ruff and type checks with mypy
	$(POETRY) run ruff check src tests scripts
	$(POETRY) run mypy src tests

format: ## Format code using Ruff
	$(POETRY) run ruff check --fix src tests scripts
	$(POETRY) run ruff format src tests scripts
	$(MAKE) clean

test: ## Execute test suite via tox (py)
	$(POETRY) run tox -e py

##@Documentation
docs-up: ## Start live Jekyll docs server (docs stack only)
	$(DOCS_COMPOSE_CMD) up --detach --build jekyll

docs-down: ## Stop docs containers
	$(DOCS_COMPOSE_CMD) down --remove-orphans

docs-logs: ## Tail Jekyll server logs
	$(DOCS_COMPOSE_CMD) logs -f jekyll

docs-build: ## One-off static docs build (does not start the live server)
	$(DOCS_COMPOSE_CMD) run --rm jekyll-build

docs-check: ## One-off link check with Lychee (run docs-build first)
	$(DOCS_COMPOSE_CMD) run --rm lychee

docs-ps: ## Show status of docs containers
	$(DOCS_COMPOSE_CMD) ps

docs-open: ## Open the local documentation site in your default browser
	@ $(POETRY) run python -c "import webbrowser; webbrowser.open('http://localhost:4000')"


##@Pipelines
pipeline-run: ## Execute sample training pipeline
	$(POETRY) run python scripts/run_pipeline.py --demo

pipeline-help: ## Show pipeline CLI help
	$(POETRY) run python scripts/run_pipeline.py --help

##@Frontend
frontend-up: ## Start frontend dev server in Docker
	$(COMPOSE_CMD) up --detach --build frontend

frontend-down: ## Stop frontend container
	$(COMPOSE_CMD) stop frontend

frontend-logs: ## Tail frontend dev server logs
	$(COMPOSE_CMD) logs -f frontend

frontend-open: ## Open the frontend in your default browser
	@ $(POETRY) run python -c "import webbrowser; webbrowser.open('http://localhost:3000')"

frontend-build: ## Run production build inside container
	$(COMPOSE_CMD) run --rm frontend npm run build

##@Backend
backend-up: ## Start FastAPI backend dev server in Docker
	$(COMPOSE_CMD) up --detach --build backend

backend-down: ## Stop backend container
	$(COMPOSE_CMD) stop backend

backend-logs: ## Tail backend dev server logs
	$(COMPOSE_CMD) logs -f backend

backend-open: ## Open the backend API docs in your default browser
	@ python3 -c "import webbrowser; webbrowser.open('http://localhost:8000/docs')"

backend-local: ## Run backend locally (no Docker); use if Docker proxy fails
	PYTHONPATH=src $(POETRY) run python -m uvicorn main_module.api.main:app --host 0.0.0.0 --port 8000 --reload

##@Frontend (local)
frontend-local: ## Run frontend locally (no Docker); requires backend first; use if Docker proxy fails
	cd src/main_module/visualization && VITE_API_URL=http://localhost:8000 npm run dev

##@Maintenance
clean: ## Remove caches and build artifacts
	@ $(MAKE) clean-pyc
	@ $(MAKE) clean-build
	@ $(MAKE) clean-test
	@ $(MAKE) clean-cache
	@ echo "Cleaned up build artifacts and caches"

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-test:
	rm -rf .tox
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache
	rm -rf coverage_*.xml
	rm -rf coverage.xml

clean-cache:
	rm -rf .mypy_cache .pytest_cache .ruff_cache dist build

precommit: ## Run all pre-commit hooks against the codebase
	$(POETRY) run pre-commit run --all-files


##@General
help: ## Show grouped make targets with descriptions
	@awk 'BEGIN { \
		FS = ":.*?##"; \
		header = "============================================"; \
		title_color = "\033[1;35m"; \
		section_color = "\033[1;34m"; \
		target_color = "\033[36m"; \
		desc_color = "\033[0;37m"; \
		muted_color = "\033[0;90m"; \
		reset = "\033[0m"; \
		print ""; \
		printf "%s%s%s\n", title_color, header, reset; \
		printf "%s   Project Command Reference   %s\n", title_color, reset; \
		printf "%s%s%s\n\n", title_color, header, reset; \
		printf "%sUse make <target> to run a task.%s\n\n", muted_color, reset; \
	} \
	/^##@/ { \
		section = substr($$0, 4); \
		printf "%s%s%s\n", section_color, section, reset; \
		next; \
	} \
	/^[a-zA-Z0-9_-]+:.*?##/ { \
		desc = $$2; \
		gsub(/^[[:space:]]+/, "", desc); \
		printf "  • %s%-18s%s %s%s%s\n", target_color, $$1, reset, desc_color, desc, reset; \
	} \
	END { \
		printf "\n%s%s%s\n\n", title_color, header, reset; \
	}' $(MAKEFILE_LIST)
