"""Causal analysis of the model degradation.

1. Decomposition: how much of the default increase comes from a different population
   (covariate shift, explained by the reference model) vs a different relationship
   between features and default (concept shift).
2. Interventions: do(feature group := reference distribution) on the latest batch and
   measure how much of the score shift disappears.
3. G-computation: adjusted effect of high credit utilization on default, before and
   after the shock, controlling for the confounders in the assumed DAG.
"""

import json
import logging

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from credit_scoring_drift import config
from credit_scoring_drift.features import add_derived_features
from credit_scoring_drift.train import REFERENCE_SCORED_PATH

log = logging.getLogger(__name__)

CAUSAL_DIR = config.REPORTS_DIR / "causal"
HIGH_UTILIZATION = 0.5
CONFOUNDERS = ["age", "log_limit", "pay_status_1", "pay_status_2", "months_delayed", "education"]
INTERVENTION_GROUPS = {
    "age (new customer profile)": ["age"],
    "pay_status (delinquency)": config.PAY_STATUS,
    "bill_amt (inflation)": config.BILL_AMT,
    "pay_amt (payment squeeze)": config.PAY_AMT,
    "limit_bal (frozen limits)": ["limit_bal"],
}


def load_batches() -> tuple[list[dict], dict[str, pd.DataFrame]]:
    manifest = json.loads((config.BATCHES_DIR / "manifest.json").read_text())
    manifest = [e for e in manifest if "upstream-bug" not in e["batch_id"]]
    frames = {e["batch_id"]: pd.read_csv(config.BATCHES_DIR / e["file"]) for e in manifest}
    return manifest, frames


def decompose(manifest, frames, model, reference: pd.DataFrame) -> pd.DataFrame:
    ref_rate = reference[config.TARGET].mean()
    calibration_offset = reference["score"].mean() - ref_rate
    rows = []
    for entry in manifest:
        df = frames[entry["batch_id"]]
        expected = model.predict_proba(df[config.MODEL_FEATURES])[:, 1].mean() - calibration_offset
        observed = df[config.TARGET].mean()
        rows.append(
            {
                "batch_id": entry["batch_id"],
                "reference_rate": ref_rate,
                "observed_rate": observed,
                "covariate_effect": expected - ref_rate,
                "concept_effect": observed - expected,
                "injected_concept_effect": entry.get("injected_concept_default", 0.0),
            }
        )
    return pd.DataFrame(rows).round(4)


def quantile_map(current: pd.Series, reference: pd.Series) -> np.ndarray:
    ranks = current.rank(pct=True, method="average").clip(0, 1)
    return np.quantile(reference.to_numpy(), ranks, method="inverted_cdf")


def interventions(df: pd.DataFrame, model, reference: pd.DataFrame) -> pd.DataFrame:
    base = model.predict_proba(df[config.MODEL_FEATURES])[:, 1].mean()
    shift = base - reference["score"].mean()
    rows = []
    for name, cols in INTERVENTION_GROUPS.items():
        treated = df.copy()
        for col in cols:
            treated[col] = quantile_map(df[col], reference[col])
        mean_score = model.predict_proba(treated[config.MODEL_FEATURES])[:, 1].mean()
        rows.append(
            {
                "intervention": f"do({name} = reference)",
                "mean_score": mean_score,
                "score_change": mean_score - base,
                "share_of_shift_removed": (base - mean_score) / shift if shift else 0.0,
            }
        )
    out = pd.DataFrame(rows).sort_values("share_of_shift_removed", ascending=False)
    return out.round(4).reset_index(drop=True)


def _design(df: pd.DataFrame) -> pd.DataFrame:
    derived = add_derived_features(df)
    out = pd.DataFrame(
        {
            "treated": (derived["utilization"] > HIGH_UTILIZATION).astype(int),
            "on_time": (df["pay_status_1"] <= 0).astype(int),
            "age": df["age"],
            "log_limit": np.log(df["limit_bal"]),
            "pay_status_1": df["pay_status_1"],
            "pay_status_2": df["pay_status_2"],
            "months_delayed": derived["months_delayed"],
            "education": df["education"],
        }
    )
    out["treated_x_on_time"] = out["treated"] * out["on_time"]
    return out


def g_computation(df: pd.DataFrame, subgroup_on_time: bool = False) -> float:
    X = _design(df)
    y = df[config.TARGET].to_numpy()
    cols = ["treated", "on_time", "treated_x_on_time", *CONFOUNDERS]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X[cols], y)
    population = X[X["on_time"] == 1] if subgroup_on_time else X
    t1, t0 = population.copy(), population.copy()
    t1["treated"], t0["treated"] = 1, 0
    t1["treated_x_on_time"] = t1["on_time"]
    t0["treated_x_on_time"] = 0
    return float(
        model.predict_proba(t1[cols])[:, 1].mean() - model.predict_proba(t0[cols])[:, 1].mean()
    )


def bootstrap_effect(df, subgroup_on_time, n_boot=100, seed=config.SEED) -> dict:
    rng = np.random.default_rng(seed)
    estimates = [
        g_computation(df.iloc[rng.integers(0, len(df), len(df))], subgroup_on_time)
        for _ in range(n_boot)
    ]
    return {
        "ate": round(g_computation(df, subgroup_on_time), 4),
        "ci_low": round(float(np.percentile(estimates, 2.5)), 4),
        "ci_high": round(float(np.percentile(estimates, 97.5)), 4),
    }


def effects_over_time(manifest, frames, reference_full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df in [("reference", reference_full)] + [
        (e["batch_id"], frames[e["batch_id"]]) for e in manifest
    ]:
        rows.append(
            {
                "dataset": name,
                **{f"all_{k}": v for k, v in bootstrap_effect(df, False).items()},
                **{f"on_time_{k}": v for k, v in bootstrap_effect(df, True).items()},
            }
        )
    return pd.DataFrame(rows)


def _plots(decomp: pd.DataFrame, interv: pd.DataFrame, effects: pd.DataFrame) -> str:
    layout = dict(template="plotly_white", height=380, xaxis_type="category")
    f1 = go.Figure()
    f1.add_bar(x=decomp["batch_id"], y=decomp["covariate_effect"], name="covariate shift")
    f1.add_bar(x=decomp["batch_id"], y=decomp["concept_effect"], name="concept shift")
    f1.add_scatter(
        x=decomp["batch_id"],
        y=decomp["injected_concept_effect"],
        name="injected concept (ground truth)",
        mode="lines+markers",
    )
    f1.update_layout(
        title="Change in default rate vs reference",
        barmode="relative",
        yaxis_tickformat=".0%",
        **layout,
    )
    f2 = go.Figure(
        go.Bar(x=interv["share_of_shift_removed"], y=interv["intervention"], orientation="h")
    )
    f2.update_layout(
        title="Share of score shift removed by each intervention (latest batch)",
        template="plotly_white",
        height=380,
        xaxis_tickformat=".0%",
        margin=dict(l=320),
    )
    f3 = go.Figure()
    for prefix, label in [("all", "all customers"), ("on_time", "on-time payers")]:
        f3.add_scatter(
            x=effects["dataset"],
            y=effects[f"{prefix}_ate"],
            name=label,
            mode="lines+markers",
            error_y=dict(
                type="data",
                symmetric=False,
                array=effects[f"{prefix}_ci_high"] - effects[f"{prefix}_ate"],
                arrayminus=effects[f"{prefix}_ate"] - effects[f"{prefix}_ci_low"],
            ),
        )
    f3.update_layout(
        title=f"Adjusted effect of utilization > {HIGH_UTILIZATION} on default (95% CI)",
        yaxis_tickformat=".0%",
        **layout,
    )
    body = "".join(f.to_html(full_html=False, include_plotlyjs=False) for f in (f1, f2, f3))
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Causal analysis</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:1.5rem}}</style>
</head><body><h1>Causal analysis of the degradation</h1>{body}</body></html>"""


def run() -> dict:
    model = joblib.load(config.MODEL_PATH)
    reference = pd.read_parquet(REFERENCE_SCORED_PATH)
    reference_full = pd.read_parquet(config.REFERENCE_PATH)
    manifest, frames = load_batches()

    decomp = decompose(manifest, frames, model, reference)
    latest = manifest[-1]["batch_id"]
    interv = interventions(frames[latest], model, reference)
    effects = effects_over_time(manifest, frames, reference_full)

    CAUSAL_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "latest_batch": latest,
        "decomposition": decomp.to_dict(orient="records"),
        "interventions": interv.to_dict(orient="records"),
        "utilization_effect": effects.to_dict(orient="records"),
    }
    (CAUSAL_DIR / "causal_analysis.json").write_text(json.dumps(result, indent=2))
    (CAUSAL_DIR / "causal_analysis.html").write_text(_plots(decomp, interv, effects))
    md = [
        "## Decomposition\n",
        decomp.to_markdown(index=False),
        f"\n\n## Interventions ({latest})\n",
        interv.to_markdown(index=False),
        "\n\n## Adjusted effect of high utilization (g-computation)\n",
        effects.to_markdown(index=False),
        "\n",
    ]
    (CAUSAL_DIR / "causal_analysis.md").write_text("\n".join(md))
    log.info("causal analysis written to %s", CAUSAL_DIR)
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run()


if __name__ == "__main__":
    main()
