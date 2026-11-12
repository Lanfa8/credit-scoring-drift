from dataclasses import asdict, dataclass

from credit_scoring_drift.drift import PSI_SEVERE, PSI_THRESHOLD, DriftResult

DATASET_DRIFT_WARNING = 0.15
DATASET_DRIFT_CRITICAL = 0.25
AUC_DROP_WARNING = 0.03
AUC_DROP_CRITICAL = 0.07
CALIBRATION_GAP_WARNING = 0.05
CALIBRATION_GAP_CRITICAL = 0.10
FOUR_FIFTHS = 0.8


@dataclass
class Alert:
    batch_id: str
    alert: str
    severity: str
    subject: str
    value: float
    threshold: float
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def _tiered(value, warning, critical) -> str | None:
    if value >= critical:
        return "critical"
    if value >= warning:
        return "warning"
    return None


def ingestion_blocked(batch_id: str, invalid_rows: int, violations: dict) -> Alert:
    top = ", ".join(f"{k} ({v})" for k, v in list(violations.items())[:3])
    return Alert(
        batch_id,
        "IngestionBlocked",
        "critical",
        "data_contract",
        float(invalid_rows),
        0.0,
        f"batch rejected by data contract, {invalid_rows} invalid rows: {top}",
    )


def evaluate(
    batch_id: str,
    drift: DriftResult,
    performance: dict[str, float] | None,
    baseline: dict[str, float],
    fairness: dict[str, dict] | None = None,
) -> list[Alert]:
    alerts: list[Alert] = []

    level = _tiered(drift.drift_share, DATASET_DRIFT_WARNING, DATASET_DRIFT_CRITICAL)
    if level:
        alerts.append(
            Alert(
                batch_id,
                "DatasetDrift",
                level,
                "dataset",
                drift.drift_share,
                DATASET_DRIFT_WARNING,
                f"{len(drift.drifted_features)} features drifted "
                f"({drift.drift_share:.0%}): {', '.join(drift.drifted_features)}",
            )
        )

    for row in drift.stats.itertuples():
        if row.drift:
            alerts.append(
                Alert(
                    batch_id,
                    "FeatureDrift",
                    "critical" if row.psi >= PSI_SEVERE else "warning",
                    row.feature,
                    row.psi,
                    PSI_THRESHOLD,
                    f"{row.feature}: PSI={row.psi:.3f}, KS p={row.ks_pvalue:.1e}, "
                    f"mean {row.ref_mean:,.2f} -> {row.cur_mean:,.2f}",
                )
            )

    level = _tiered(drift.prediction_psi, PSI_THRESHOLD, PSI_SEVERE)
    if level:
        alerts.append(
            Alert(
                batch_id,
                "PredictionDrift",
                level,
                "score",
                drift.prediction_psi,
                PSI_THRESHOLD,
                f"score distribution shifted, PSI={drift.prediction_psi:.3f}",
            )
        )

    if performance:
        auc_drop = baseline["roc_auc"] - performance["roc_auc"]
        level = _tiered(auc_drop, AUC_DROP_WARNING, AUC_DROP_CRITICAL)
        if level:
            alerts.append(
                Alert(
                    batch_id,
                    "ModelPerformanceDegraded",
                    level,
                    "roc_auc",
                    round(auc_drop, 4),
                    AUC_DROP_WARNING,
                    f"AUC {performance['roc_auc']:.3f} vs baseline {baseline['roc_auc']:.3f}",
                )
            )

        gap = performance["actual_default_rate"] - performance["mean_score"]
        level = _tiered(gap, CALIBRATION_GAP_WARNING, CALIBRATION_GAP_CRITICAL)
        if level:
            alerts.append(
                Alert(
                    batch_id,
                    "ConceptDriftSuspected",
                    level,
                    "calibration",
                    round(gap, 4),
                    CALIBRATION_GAP_WARNING,
                    f"observed default {performance['actual_default_rate']:.1%} vs "
                    f"expected {performance['mean_score']:.1%}: P(default|x) changed",
                )
            )

    for attribute, info in (fairness or {}).items():
        if info["disparate_impact"] < FOUR_FIFTHS:
            alerts.append(
                Alert(
                    batch_id,
                    "FairnessViolation",
                    "warning",
                    attribute,
                    info["disparate_impact"],
                    FOUR_FIFTHS,
                    f"approval rate ratio for {attribute} = {info['disparate_impact']:.2f}",
                )
            )

    return alerts
