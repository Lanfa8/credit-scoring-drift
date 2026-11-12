import hashlib
import json
import logging
from datetime import UTC, datetime

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from scipy.stats import ks_2samp
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

from credit_scoring_drift import config
from credit_scoring_drift.features import add_derived_features, age_band

log = logging.getLogger(__name__)

EXPERIMENT = "credit-scoring-baseline"
REGISTERED_MODEL = "credit-scoring-xgb"
MODEL_REPORT_DIR = config.REPORTS_DIR / "model"
REFERENCE_SCORED_PATH = config.PROCESSED_DIR / "reference_scored.parquet"

PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "eval_metric": "logloss",
    "random_state": config.SEED,
    "n_jobs": -1,
}


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("features", FunctionTransformer(add_derived_features)),
            ("clf", XGBClassifier(**PARAMS)),
        ]
    )


def best_f1_threshold(y_true, scores) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.argmax(f1[:-1])])


def classification_metrics(y_true, scores, threshold: float) -> dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "ks": float(ks_2samp(scores[y_true == 1], scores[y_true == 0]).statistic),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, scores),
        "predicted_default_rate": float(y_pred.mean()),
        "actual_default_rate": float(np.mean(y_true)),
    }


def fairness_report(df: pd.DataFrame, scores, threshold: float) -> dict:
    approved = pd.Series(scores < threshold, index=df.index)
    groups = {
        "sex": df["sex"].map({1: "male", 2: "female"}),
        "age_band": age_band(df["age"]).astype(str),
    }
    report = {}
    for name, group in groups.items():
        rates = approved.groupby(group).mean()
        report[name] = {
            "approval_rate": rates.round(4).to_dict(),
            "disparate_impact": round(float(rates.min() / rates.max()), 4),
            "passes_four_fifths_rule": bool(rates.min() / rates.max() >= 0.8),
        }
    return report


def data_fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values).hexdigest()[:12]


def train():
    reference = pd.read_parquet(config.REFERENCE_PATH)
    train_df, test_df = train_test_split(
        reference, test_size=0.25, stratify=reference[config.TARGET], random_state=config.SEED
    )
    X_train, y_train = train_df[config.MODEL_FEATURES], train_df[config.TARGET].to_numpy()
    X_test, y_test = test_df[config.MODEL_FEATURES], test_df[config.TARGET].to_numpy()

    cv = StratifiedKFold(5, shuffle=True, random_state=config.SEED)
    oof = cross_val_predict(build_model(), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = best_f1_threshold(y_train, oof)

    model = build_model().fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, scores, threshold)
    fairness = fairness_report(test_df, scores, threshold)

    clf = model.named_steps["clf"]
    feature_names = list(add_derived_features(X_test.head()).columns)
    importance = dict(
        sorted(
            zip(feature_names, [round(float(v), 4) for v in clf.feature_importances_], strict=True),
            key=lambda kv: -kv[1],
        )
    )

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"xgb-baseline-{datetime.now():%Y%m%d-%H%M}") as run:
        mlflow.log_params({**PARAMS, "threshold": round(threshold, 4)})
        mlflow.log_params({"train_rows": len(train_df), "test_rows": len(test_df)})
        mlflow.set_tags(
            {
                "reference_fingerprint": data_fingerprint(reference),
                "excluded_sensitive_features": ",".join(config.SENSITIVE_COLS),
            }
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_metrics({f"di_{k}": v["disparate_impact"] for k, v in fairness.items()})
        mlflow.log_dict(fairness, "fairness.json")
        mlflow.log_dict(importance, "feature_importance.json")
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=REGISTERED_MODEL,
            signature=infer_signature(X_test, scores),
            input_example=X_test.head(3),
            skops_trusted_types=[
                "credit_scoring_drift.features.add_derived_features",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )
        run_id = run.info.run_id

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    metadata = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mlflow_run_id": run_id,
        "registered_model": REGISTERED_MODEL,
        "threshold": threshold,
        "features": config.MODEL_FEATURES,
        "excluded_features": config.SENSITIVE_COLS,
        "reference_fingerprint": data_fingerprint(reference),
        "metrics": metrics,
        "fairness": fairness,
        "feature_importance": importance,
    }
    config.MODEL_META_PATH.write_text(json.dumps(metadata, indent=2, default=float))
    MODEL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_REPORT_DIR / "baseline.json").write_text(json.dumps(metadata, indent=2, default=float))

    scored = test_df.assign(score=scores, prediction=(scores >= threshold).astype(int))
    scored.to_parquet(REFERENCE_SCORED_PATH, index=False)

    log.info("baseline trained: %s", {k: round(v, 4) for k, v in metrics.items()})
    log.info("fairness: %s", {k: v["disparate_impact"] for k, v in fairness.items()})
    return metadata


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    train()


if __name__ == "__main__":
    main()
