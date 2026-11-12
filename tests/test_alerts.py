import pandas as pd

from credit_scoring_drift import alerts
from credit_scoring_drift.drift import DriftResult

BASELINE = {"roc_auc": 0.78}


def _drift(share=0.0, drifted=(), prediction_psi=0.0):
    stats = pd.DataFrame(
        [
            {
                "feature": f,
                "psi": 0.3,
                "ks_pvalue": 1e-10,
                "ref_mean": 1.0,
                "cur_mean": 2.0,
                "drift": True,
            }
            for f in drifted
        ],
        columns=["feature", "psi", "ks_pvalue", "ref_mean", "cur_mean", "drift"],
    )
    return DriftResult("b", stats, share, list(drifted), prediction_psi, None, share)


def _perf(auc, actual, expected):
    return {"roc_auc": auc, "actual_default_rate": actual, "mean_score": expected}


def test_stable_batch_raises_nothing():
    assert alerts.evaluate("b", _drift(), _perf(0.78, 0.22, 0.22), BASELINE) == []


def test_degraded_batch_raises_expected_alerts():
    raised = alerts.evaluate(
        "b",
        _drift(share=0.3, drifted=["age", "utilization"], prediction_psi=0.3),
        _perf(0.65, 0.45, 0.30),
        BASELINE,
        {"sex": {"disparate_impact": 0.7}},
    )
    names = {(a.alert, a.severity) for a in raised}
    assert ("DatasetDrift", "critical") in names
    assert ("FeatureDrift", "critical") in names
    assert ("PredictionDrift", "critical") in names
    assert ("ModelPerformanceDegraded", "critical") in names
    assert ("ConceptDriftSuspected", "critical") in names
    assert ("FairnessViolation", "warning") in names


def test_small_auc_drop_is_warning():
    raised = alerts.evaluate("b", _drift(), _perf(0.745, 0.22, 0.22), BASELINE)
    assert [(a.alert, a.severity) for a in raised] == [("ModelPerformanceDegraded", "warning")]
