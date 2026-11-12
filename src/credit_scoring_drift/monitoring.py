import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime

import requests
from prometheus_client import CollectorRegistry, Gauge, ProcessCollector, push_to_gateway
from prometheus_client.exposition import write_to_textfile

from credit_scoring_drift import config

APP = "credit-pipeline"
METRICS_DIR = config.REPORTS_DIR / "metrics"

_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _RESERVED})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    def __init__(self):
        super().__init__("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        batch = getattr(record, "batch_id", None)
        return f"{line} [{batch}]" if batch else line


class LokiHandler(logging.Handler):
    def __init__(self, url: str, labels: dict[str, str], batch_size: int = 20):
        super().__init__()
        self.url = url.rstrip("/") + "/loki/api/v1/push"
        self.labels = labels
        self.batch_size = batch_size
        self.buffer: list[tuple[str, str, str]] = []
        self.enabled = True

    def emit(self, record: logging.LogRecord) -> None:
        if not self.enabled:
            return
        ts = str(int(record.created * 1e9))
        self.buffer.append((record.levelname.lower(), ts, self.format(record)))
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer or not self.enabled:
            return
        streams: dict[str, list] = {}
        for level, ts, line in self.buffer:
            streams.setdefault(level, []).append([ts, line])
        body = {
            "streams": [
                {"stream": {**self.labels, "level": level}, "values": values}
                for level, values in streams.items()
            ]
        }
        self.buffer = []
        try:
            requests.post(self.url, json=body, timeout=2).raise_for_status()
        except requests.RequestException:
            self.enabled = False

    def close(self) -> None:
        self.flush()
        super().close()


def setup_logging(level: int = logging.INFO) -> None:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ConsoleFormatter())
    root.addHandler(console)

    jsonl = logging.FileHandler(config.LOGS_DIR / "pipeline.jsonl")
    jsonl.setFormatter(JsonFormatter())
    root.addHandler(jsonl)

    loki = LokiHandler(config.LOKI_URL, {"app": APP})
    loki.setFormatter(JsonFormatter())
    root.addHandler(loki)

    for noisy in ("mlflow", "urllib3", "alembic", "matplotlib", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class _LazyGauge:
    """Unlabelled gauges are exported as 0 as soon as they exist; create them on first
    use so a blocked batch does not report, e.g., drift_share=0."""

    def __init__(self, name: str, doc: str, registry: CollectorRegistry):
        self.args = (name, doc)
        self.registry = registry
        self.gauge: Gauge | None = None

    def _get(self) -> Gauge:
        if self.gauge is None:
            self.gauge = Gauge(*self.args, registry=self.registry)
        return self.gauge

    def set(self, value: float) -> None:
        self._get().set(value)

    def set_to_current_time(self) -> None:
        self._get().set_to_current_time()


class BatchMetrics:
    def __init__(self, batch_id: str):
        self.batch_id = batch_id
        self.registry = CollectorRegistry()
        ProcessCollector(registry=self.registry)
        r = self.registry
        self.task_success = Gauge(
            "credit_pipeline_task_success", "1 if the task succeeded", ["task"], registry=r
        )
        self.task_duration = Gauge(
            "credit_pipeline_task_duration_seconds", "Task wall time", ["task"], registry=r
        )
        self.batch_status = _LazyGauge(
            "credit_pipeline_batch_status", "1 accepted, 0 blocked by contract", r
        )
        self.rows = Gauge("credit_pipeline_rows", "Rows per stage", ["stage"], registry=r)
        self.violations = Gauge(
            "credit_contract_violations", "Contract violations by check", ["check"], registry=r
        )
        self.drift_share = _LazyGauge("credit_drift_share", "Share of drifted features", r)
        self.drifted = _LazyGauge("credit_drifted_features", "Number of drifted features", r)
        self.feature_psi = Gauge("credit_feature_psi", "PSI per feature", ["feature"], registry=r)
        self.feature_ks = Gauge(
            "credit_feature_ks_statistic", "KS statistic per feature", ["feature"], registry=r
        )
        self.prediction_psi = _LazyGauge("credit_prediction_psi", "PSI of model scores", r)
        self.target_psi = _LazyGauge("credit_target_psi", "PSI of observed default", r)
        self.model_metric = Gauge(
            "credit_model_metric", "Model performance on labelled batch", ["metric"], registry=r
        )
        self.auc_drop = _LazyGauge("credit_model_auc_drop", "Baseline AUC minus batch AUC", r)
        self.default_rate = Gauge(
            "credit_default_rate",
            "Default rates (predicted/actual/mean score)",
            ["kind"],
            registry=r,
        )
        self.fairness = Gauge(
            "credit_fairness_disparate_impact", "Approval rate ratio", ["attribute"], registry=r
        )
        self.latency = _LazyGauge(
            "credit_inference_latency_seconds", "Scoring latency for the batch", r
        )
        self.throughput = _LazyGauge("credit_inference_rows_per_second", "Scoring throughput", r)
        self.alerts = Gauge(
            "credit_alerts", "Alerts raised for the batch", ["severity"], registry=r
        )
        self.alert_active = Gauge(
            "credit_alert_active", "Alert raised", ["alert", "severity", "subject"], registry=r
        )
        self.last_run = _LazyGauge(
            "credit_pipeline_last_run_timestamp_seconds", "Last processing time", r
        )

    @contextmanager
    def track(self, task: str):
        start = time.perf_counter()
        try:
            yield
        except Exception:
            self.task_success.labels(task).set(0)
            raise
        else:
            self.task_success.labels(task).set(1)
        finally:
            self.task_duration.labels(task).set(time.perf_counter() - start)

    def publish(self) -> bool:
        self.last_run.set_to_current_time()
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        write_to_textfile(str(METRICS_DIR / f"{self.batch_id}.prom"), self.registry)
        try:
            push_to_gateway(
                config.PUSHGATEWAY_URL,
                job="credit_pipeline",
                grouping_key={"batch": self.batch_id},
                registry=self.registry,
                timeout=3,
            )
        except OSError:
            return False
        return True
