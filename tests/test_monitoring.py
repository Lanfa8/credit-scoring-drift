import json
import logging

from prometheus_client import generate_latest

from credit_scoring_drift.monitoring import BatchMetrics, JsonFormatter


def test_blocked_batch_does_not_export_drift_metrics():
    metrics = BatchMetrics("blocked")
    metrics.batch_status.set(0)
    exported = generate_latest(metrics.registry).decode()
    assert "credit_pipeline_batch_status 0.0" in exported
    assert "credit_drift_share" not in exported
    assert "credit_model_auc_drop" not in exported


def test_track_records_failure():
    metrics = BatchMetrics("b")
    try:
        with metrics.track("score"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    exported = generate_latest(metrics.registry).decode()
    assert 'credit_pipeline_task_success{task="score"} 0.0' in exported


def test_json_formatter_keeps_extra_fields():
    record = logging.makeLogRecord(
        {"msg": "batch accepted", "levelname": "INFO", "batch_id": "2025-01", "rows": 10}
    )
    payload = json.loads(JsonFormatter().format(record))
    assert payload["batch_id"] == "2025-01"
    assert payload["rows"] == 10
    assert payload["message"] == "batch accepted"
