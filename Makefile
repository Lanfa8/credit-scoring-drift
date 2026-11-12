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
