.PHONY: setup dev test style build clean infra-up infra-down db-load dbt-run eval help

## Install all dependencies (frontend + backend)
setup:
	@echo "==> Installing frontend dependencies..."
	cd frontend && npm install
	@echo "==> Installing backend dependencies..."
	cd backend && uv sync

## Run frontend + backend dev servers concurrently
dev:
	@trap 'kill 0' EXIT; \
	cd backend && uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload & \
	cd frontend && npm run dev & \
	wait

## Run all tests (frontend + backend)
test:
	cd frontend && npm run test
	cd backend && uv run pytest

## Format + lint all code
style:
	cd frontend && npm run format && npm run lint
	cd backend && uv run ruff check . --fix && uv run ruff format .

## Production build of frontend
build:
	cd frontend && npm run build

## Remove build artifacts, caches, node_modules
clean:
	rm -rf frontend/dist frontend/node_modules
	rm -rf backend/__pycache__ backend/.pytest_cache backend/.venv backend/.ruff_cache
	rm -rf .venv __pycache__ .ruff_cache

## Start docker-compose services (MySQL, backend, Prometheus, Grafana, Sandbox)
infra-up:
	docker compose up -d

## Stop docker-compose services
infra-down:
	docker compose down

## Load MySQL dump into local database (starts MySQL, waits for health, streams dump)
db-load:
	docker compose up -d mysql
	@echo "==> Waiting for MySQL to be ready..."
	@until docker exec $$(docker compose ps -q mysql 2>/dev/null) mysqladmin ping -h localhost -uroot --silent 2>/dev/null; do sleep 2; done
	@echo "==> Loading db-dump.sql (this takes a few minutes)..."
	docker exec -i $$(docker compose ps -q mysql 2>/dev/null) mysql -uroot < db-dump.sql
	@echo "==> Done."

## Run dbt models (MySQL -> DuckDB transforms)
dbt-run:
	cd backend && uv run --group dev dbt run --project-dir ../dbt --profiles-dir ../dbt

## Run the NL-to-SQL eval harness (schema check only, CI-safe)
eval:
	python eval/runner.py --check

## Show this help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
