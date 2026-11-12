import argparse
import json
import logging
import time
from datetime import UTC, datetime

import joblib
import mlflow
import pandas as pd

from credit_scoring_drift import alerts, config, drift
from credit_scoring_drift.ingest import IngestionBlocked, ingest
from credit_scoring_drift.monitoring import BatchMetrics, setup_logging
from credit_scoring_drift.train import (
    REFERENCE_SCORED_PATH,
    classification_metrics,
    fairness_report,
)

log = logging.getLogger("credit_pipeline")

EXPERIMENT = "credit-monitoring"
SUMMARY_PATH = config.REPORTS_DIR / "pipeline_summary.json"
ALERTS_PATH = config.REPORTS_DIR / "alerts.json"


def _log_alerts(batch_alerts: list[alerts.Alert]) -> None:
    for a in batch_alerts:
        level = logging.ERROR if a.severity == "critical" else logging.WARNING
        log.log(
            level,
            f"ALERT {a.alert}: {a.message}",
            extra={"batch_id": a.batch_id, "alert": a.to_dict()},
        )


def _record_alert_metrics(metrics: BatchMetrics, batch_alerts: list[alerts.Alert]) -> None:
    for severity in ("warning", "critical"):
        metrics.alerts.labels(severity).set(sum(a.severity == severity for a in batch_alerts))
    for a in batch_alerts:
        metrics.alert_active.labels(a.alert, a.severity, a.subject).set(a.value)


def process_blocked(entry, exc: IngestionBlocked, metrics: BatchMetrics) -> dict:
    result = exc.result
    violations = result.summary_by_check()
    metrics.batch_status.set(0)
    metrics.rows.labels("received").set(result.n_rows)
    metrics.rows.labels("accepted").set(0)
    metrics.rows.labels("rejected").set(result.n_rows)
    for check, count in violations.items():
        metrics.violations.labels(check).set(count)

    alert = alerts.ingestion_blocked(entry["batch_id"], result.failed_rows, violations)
    _log_alerts([alert])
    _record_alert_metrics(metrics, [alert])

    mlflow.set_tag("status", "blocked")
    mlflow.log_metrics({"rows_received": result.n_rows, "invalid_rows": result.failed_rows})
    mlflow.log_dict(violations, "contract_violations.json")
    mlflow.log_artifact(str(config.REPORTS_DIR / "quality" / f"{entry['batch_id']}.html"))
    return {
        **entry,
        "status": "blocked",
        "rows_received": result.n_rows,
        "invalid_rows": result.failed_rows,
        "violations": violations,
        "alerts": [alert.to_dict()],
    }


def process_batch(entry, df, model, meta, reference, metrics: BatchMetrics) -> dict:
    batch_id = entry["batch_id"]
    threshold = meta["threshold"]
    metrics.batch_status.set(1)
    metrics.rows.labels("received").set(len(df))
    metrics.rows.labels("accepted").set(len(df))
    metrics.rows.labels("rejected").set(0)

    with metrics.track("score"):
        start = time.perf_counter()
        df["score"] = model.predict_proba(df[config.MODEL_FEATURES])[:, 1]
        df["prediction"] = (df["score"] >= threshold).astype(int)
        latency = time.perf_counter() - start
    metrics.latency.set(latency)
    metrics.throughput.set(len(df) / latency)
    log.info(
        "batch scored",
        extra={"batch_id": batch_id, "rows": len(df), "latency_s": round(latency, 4)},
    )

    with metrics.track("drift"):
        result = drift.detect(reference, df, batch_id)
    metrics.drift_share.set(result.drift_share)
    metrics.drifted.set(len(result.drifted_features))
    metrics.prediction_psi.set(result.prediction_psi)
    for row in result.stats.itertuples():
        metrics.feature_psi.labels(row.feature).set(row.psi)
        metrics.feature_ks.labels(row.feature).set(row.ks_statistic)
    log.info(
        "drift analysis done",
        extra={
            "batch_id": batch_id,
            "drift_share": result.drift_share,
            "drifted_features": result.drifted_features,
            "prediction_psi": result.prediction_psi,
        },
    )

    performance, fairness = None, None
    with metrics.track("evaluate"):
        if df[config.TARGET].notna().all():
            y = df[config.TARGET].to_numpy()
            performance = classification_metrics(y, df["score"].to_numpy(), threshold)
            performance["mean_score"] = float(df["score"].mean())
            fairness = fairness_report(df, df["score"].to_numpy(), threshold)
            metrics.target_psi.set(result.target_psi or 0)
            for name in ("roc_auc", "ks", "f1", "precision", "recall", "brier"):
                metrics.model_metric.labels(name).set(performance[name])
            metrics.auc_drop.set(meta["metrics"]["roc_auc"] - performance["roc_auc"])
            metrics.default_rate.labels("actual").set(performance["actual_default_rate"])
            metrics.default_rate.labels("predicted").set(performance["predicted_default_rate"])
            metrics.default_rate.labels("mean_score").set(performance["mean_score"])
            for attribute, info in fairness.items():
                metrics.fairness.labels(attribute).set(info["disparate_impact"])

    batch_alerts = alerts.evaluate(batch_id, result, performance, meta["metrics"], fairness)
    _log_alerts(batch_alerts)
    _record_alert_metrics(metrics, batch_alerts)

    mlflow.set_tag("status", "processed")
    mlflow.log_metrics(
        {
            "drift_share": result.drift_share,
            "drifted_features": len(result.drifted_features),
            "prediction_psi": result.prediction_psi,
            "alerts_critical": sum(a.severity == "critical" for a in batch_alerts),
            "alerts_warning": sum(a.severity == "warning" for a in batch_alerts),
            "inference_latency_s": latency,
            **{f"psi_{r.feature}": r.psi for r in result.stats.itertuples()},
            **({f"batch_{k}": v for k, v in performance.items()} if performance else {}),
        }
    )
    mlflow.log_artifact(str(result.report_path), "evidently")
    mlflow.log_artifact(str(config.REPORTS_DIR / "quality" / f"{batch_id}.html"), "quality")
    mlflow.log_dict([a.to_dict() for a in batch_alerts], "alerts.json")

    return {
        **entry,
        "status": "processed",
        "rows_received": len(df),
        "drift": {k: v for k, v in result.to_dict().items() if k != "features"},
        "feature_psi": dict(zip(result.stats["feature"], result.stats["psi"], strict=True)),
        "performance": performance,
        "fairness": fairness,
        "alerts": [a.to_dict() for a in batch_alerts],
    }


def run(only: list[str] | None = None) -> list[dict]:
    setup_logging()
    manifest = json.loads((config.BATCHES_DIR / "manifest.json").read_text())
    if only:
        manifest = [e for e in manifest if e["batch_id"] in only]

    model = joblib.load(config.MODEL_PATH)
    meta = json.loads(config.MODEL_META_PATH.read_text())
    reference = pd.read_parquet(REFERENCE_SCORED_PATH)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    log.info("pipeline started", extra={"batches": [e["batch_id"] for e in manifest]})

    results = []
    for entry in manifest:
        batch_id = entry["batch_id"]
        metrics = BatchMetrics(batch_id)
        log.info("processing batch", extra={"batch_id": batch_id, "period": entry["period"]})
        with mlflow.start_run(run_name=batch_id):
            mlflow.log_params(
                {
                    "batch_id": batch_id,
                    "period": entry["period"],
                    "model_run": meta["mlflow_run_id"],
                }
            )
            try:
                with metrics.track("ingest"):
                    df = ingest(config.BATCHES_DIR / entry["file"], batch_id)
            except IngestionBlocked as exc:
                record = process_blocked(entry, exc, metrics)
            else:
                record = process_batch(entry, df, model, meta, reference, metrics)

        pushed = metrics.publish()
        if not pushed:
            log.warning("pushgateway unreachable, metrics kept in reports/metrics")
        results.append(record)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": {k: meta[k] for k in ("mlflow_run_id", "threshold", "metrics", "fairness")},
        "batches": results,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=float))
    all_alerts = [a for r in results for a in r["alerts"]]
    ALERTS_PATH.write_text(json.dumps(all_alerts, indent=2, default=float))
    log.info(
        "pipeline finished",
        extra={
            "processed": sum(r["status"] == "processed" for r in results),
            "blocked": sum(r["status"] == "blocked" for r in results),
            "alerts": len(all_alerts),
        },
    )
    logging.shutdown()
    return results


def main():
    parser = argparse.ArgumentParser(description="Run the monitoring pipeline over batches")
    parser.add_argument("--only", nargs="*", help="batch ids to process")
    args = parser.parse_args()
    run(args.only)


if __name__ == "__main__":
    main()
