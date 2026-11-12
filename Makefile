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
