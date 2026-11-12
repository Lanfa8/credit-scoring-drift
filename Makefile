PY := uv run python

.PHONY: setup lint test

setup:
	uv sync

lint:
	uv run ruff check src tests

test:
	uv run pytest -q

.PHONY: data
data:
	$(PY) -m credit_scoring_drift.data

.PHONY: validate train
validate:
	$(PY) -m credit_scoring_drift.ingest $(BATCH)

train:
	$(PY) -m credit_scoring_drift.train

.PHONY: simulate
simulate:
	$(PY) -m credit_scoring_drift.simulate

.PHONY: pipeline
pipeline:
	$(PY) -m credit_scoring_drift.pipeline

.PHONY: up down mlflow-ui
up:
	docker compose up -d

down:
	docker compose down

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

.PHONY: dashboard monitor
dashboard:
	$(PY) -m credit_scoring_drift.dashboard

monitor: pipeline dashboard

.PHONY: causal
causal:
	$(PY) -m credit_scoring_drift.causal
