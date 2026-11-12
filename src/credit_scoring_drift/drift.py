import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.legacy.calculations.stattests.registry import StatTest, register_stattest
from evidently.legacy.core import ColumnType
from evidently.presets import ClassificationPreset, DataDriftPreset
from scipy.stats import ks_2samp

from credit_scoring_drift import config
from credit_scoring_drift.features import add_derived_features

log = logging.getLogger(__name__)

DRIFT_DIR = config.REPORTS_DIR / "drift"
PSI_THRESHOLD = 0.1
PSI_SEVERE = 0.25
DATASET_DRIFT_SHARE = 0.25
KS_ALPHA = 0.01
CATEGORICAL = ["education", *config.PAY_STATUS]
NUMERICAL = [c for c in config.MONITORED_FEATURES if c not in CATEGORICAL]


def psi(reference: pd.Series, current: pd.Series, bins: int = 10, eps: float = 1e-4) -> float:
    reference, current = reference.dropna(), current.dropna()
    if reference.nunique() <= 15:
        cats = sorted(set(reference.unique()) | set(current.unique()))
        ref_pct = reference.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        cur_pct = current.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_pct = np.histogram(reference, edges)[0] / len(reference)
        cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _quantile_psi(reference, current, feature_type, threshold):
    value = psi(reference, current)
    return value, value >= threshold


# Evidently's built-in PSI uses equal-width bins, which hides shifts in long-tailed
# monetary features; quantile bins from the reference are the usual credit-risk practice.
quantile_psi_test = StatTest(
    name="quantile_psi",
    display_name="PSI (quantile bins)",
    allowed_feature_types=[ColumnType.Numerical, ColumnType.Categorical],
    default_threshold=PSI_THRESHOLD,
)
register_stattest(quantile_psi_test, _quantile_psi)


def severity(value: float) -> str:
    if value >= PSI_SEVERE:
        return "significant"
    if value >= PSI_THRESHOLD:
        return "moderate"
    return "stable"


def statistical_tests(reference: pd.DataFrame, current: pd.DataFrame, columns) -> pd.DataFrame:
    rows = []
    for col in columns:
        ks = ks_2samp(reference[col].dropna(), current[col].dropna())
        value = psi(reference[col], current[col])
        rows.append(
            {
                "feature": col,
                "psi": round(value, 4),
                "ks_statistic": round(float(ks.statistic), 4),
                "ks_pvalue": float(ks.pvalue),
                "ref_mean": round(float(reference[col].mean()), 2),
                "cur_mean": round(float(current[col].mean()), 2),
                "psi_level": severity(value),
                "drift": bool(value >= PSI_THRESHOLD and ks.pvalue < KS_ALPHA),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


@dataclass
class DriftResult:
    batch_id: str
    stats: pd.DataFrame
    drift_share: float
    drifted_features: list[str]
    prediction_psi: float
    target_psi: float | None
    evidently_drift_share: float
    evidently_psi: dict[str, float] = field(default_factory=dict)
    performance: dict[str, float] = field(default_factory=dict)
    report_path: Path | None = None

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "drift_share": self.drift_share,
            "drifted_features": self.drifted_features,
            "prediction_psi": self.prediction_psi,
            "target_psi": self.target_psi,
            "evidently_drift_share": self.evidently_drift_share,
            "performance": self.performance,
            "report": str(self.report_path) if self.report_path else None,
            "features": self.stats.to_dict(orient="records"),
        }


def _evidently_report(reference: pd.DataFrame, current: pd.DataFrame, has_target: bool):
    definition = DataDefinition(
        numerical_columns=NUMERICAL,
        categorical_columns=CATEGORICAL,
        classification=[BinaryClassification(target=config.TARGET, prediction_probas="score")]
        if has_target
        else None,
    )
    metrics = [
        DataDriftPreset(
            method="quantile_psi", threshold=PSI_THRESHOLD, drift_share=DATASET_DRIFT_SHARE
        )
    ]
    if has_target:
        metrics.append(ClassificationPreset())
    cols = [*config.MONITORED_FEATURES, "score", *([config.TARGET] if has_target else [])]
    report = Report(metrics, include_tests=True)
    return report.run(
        Dataset.from_pandas(current[cols], data_definition=definition),
        Dataset.from_pandas(reference[cols], data_definition=definition),
    )


def _parse_snapshot(snapshot) -> tuple[float, dict[str, float]]:
    share, per_column = 0.0, {}
    for metric in snapshot.dict()["metrics"]:
        cfg = metric["config"]
        kind = cfg.get("type", "")
        if kind.endswith("DriftedColumnsCount"):
            share = float(metric["value"]["share"])
        elif kind.endswith("ValueDrift"):
            per_column[cfg["column"]] = round(float(metric["value"]), 4)
    return share, per_column


def with_derived(df: pd.DataFrame) -> pd.DataFrame:
    derived = add_derived_features(df)[config.DERIVED_FEATURES]
    return df.drop(columns=config.DERIVED_FEATURES, errors="ignore").join(derived)


def detect(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    batch_id: str,
    out_dir: Path = DRIFT_DIR,
) -> DriftResult:
    has_target = config.TARGET in current and current[config.TARGET].notna().all()
    reference, current = with_derived(reference), with_derived(current)

    stats = statistical_tests(reference, current, config.MONITORED_FEATURES)
    drifted = stats.loc[stats["drift"], "feature"].tolist()

    snapshot = _evidently_report(reference, current, has_target)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{batch_id}.html"
    snapshot.save_html(str(report_path))
    ev_share, ev_psi = _parse_snapshot(snapshot)

    stats["evidently_psi"] = stats["feature"].map(ev_psi)
    stats.to_csv(out_dir / f"{batch_id}_stats.csv", index=False)

    result = DriftResult(
        batch_id=batch_id,
        stats=stats,
        drift_share=round(len(drifted) / len(stats), 4),
        drifted_features=drifted,
        prediction_psi=round(psi(reference["score"], current["score"]), 4),
        target_psi=round(psi(reference[config.TARGET], current[config.TARGET]), 4)
        if has_target
        else None,
        evidently_drift_share=round(ev_share, 4),
        evidently_psi=ev_psi,
        report_path=report_path,
    )
    (out_dir / f"{batch_id}.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result
