import html
import json

import pandas as pd
import plotly.graph_objects as go

from credit_scoring_drift import config
from credit_scoring_drift.alerts import AUC_DROP_CRITICAL
from credit_scoring_drift.drift import PSI_SEVERE, PSI_THRESHOLD
from credit_scoring_drift.pipeline import SUMMARY_PATH

OUT_PATH = config.REPORTS_DIR / "monitoring_dashboard.html"
LAYOUT = dict(
    template="plotly_white",
    height=360,
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis_type="category",
)


def _fig_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build(summary: dict | None = None) -> str:
    summary = summary or json.loads(SUMMARY_PATH.read_text())
    baseline = summary["model"]["metrics"]
    batches = summary["batches"]
    ok = [b for b in batches if b["status"] == "processed"]
    periods = [b["batch_id"] for b in ok]

    auc = go.Figure()
    auc.add_scatter(
        x=periods, y=[b["performance"]["roc_auc"] for b in ok], mode="lines+markers", name="AUC"
    )
    auc.add_hline(y=baseline["roc_auc"], line_dash="dash", annotation_text="baseline")
    auc.add_hline(
        y=baseline["roc_auc"] - AUC_DROP_CRITICAL,
        line_dash="dot",
        line_color="red",
        annotation_text="critical",
    )
    auc.update_layout(title="ROC AUC per batch", **LAYOUT)

    rates = go.Figure()
    for key, label in [
        ("actual_default_rate", "observed default"),
        ("mean_score", "expected by model"),
        ("predicted_default_rate", "flagged as risky"),
    ]:
        rates.add_bar(x=periods, y=[b["performance"][key] for b in ok], name=label)
    rates.update_layout(
        title="Default rate: observed vs expected", yaxis_tickformat=".0%", **LAYOUT
    )

    share = go.Figure(
        go.Bar(x=periods, y=[b["drift"]["drift_share"] for b in ok], name="drifted features share")
    )
    share.add_scatter(
        x=periods,
        y=[b["drift"]["prediction_psi"] for b in ok],
        name="score PSI",
        mode="lines+markers",
    )
    share.update_layout(title="Share of drifted features and score PSI", **LAYOUT)

    psi = pd.DataFrame({b["batch_id"]: b["feature_psi"] for b in ok})
    psi = psi.loc[psi.max(axis=1).sort_values(ascending=False).index]
    heat = go.Figure(
        go.Heatmap(
            z=psi.values,
            x=psi.columns,
            y=psi.index,
            colorscale=[[0, "#2da44e"], [PSI_THRESHOLD / 0.7, "#f2cc60"], [1, "#cf222e"]],
            zmin=0,
            zmax=0.7,
            text=psi.round(3).values,
            texttemplate="%{text}",
        )
    )
    heat.update_layout(
        title=f"PSI per feature (>= {PSI_THRESHOLD} moderate, >= {PSI_SEVERE} significant)",
        **{**LAYOUT, "height": 720},
    )
    heat.update_yaxes(autorange="reversed")

    rows = []
    for b in batches:
        crit = sum(a["severity"] == "critical" for a in b["alerts"])
        warn = sum(a["severity"] == "warning" for a in b["alerts"])
        bid = b["batch_id"]
        if b["status"] == "blocked":
            cells = f"<td class=bad>BLOCKED</td><td>{b['rows_received']}</td><td>-</td><td>-</td>"
        else:
            cells = (
                f"<td class=good>processed</td><td>{b['rows_received']}</td>"
                f"<td>{b['drift']['drift_share']:.0%}</td>"
                f"<td>{b['performance']['roc_auc']:.3f}</td>"
            )
        links = f"<a href='quality/{bid}.html'>contract</a>"
        if b["status"] == "processed":
            links += f" &middot; <a href='drift/{bid}.html'>evidently</a>"
        rows.append(
            f"<tr><td>{bid}</td>{cells}<td>{crit}</td><td>{warn}</td><td>{links}</td>"
            f"<td class=note>{html.escape(b['notes'])}</td></tr>"
        )

    alert_rows = "".join(
        f"<tr class={a['severity']}><td>{a['batch_id']}</td><td>{a['severity']}</td>"
        f"<td>{a['alert']}</td><td>{html.escape(a['message'])}</td></tr>"
        for b in batches
        for a in b["alerts"]
        if a["severity"] == "critical" or a["alert"] != "FeatureDrift"
    )

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Credit scoring monitoring</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{{font-family:system-ui,sans-serif;margin:0 auto;padding:1.5rem;max-width:1200px;color:#1f2328}}
table{{border-collapse:collapse;width:100%;font-size:.9rem;margin-bottom:1.5rem}}
td,th{{border-bottom:1px solid #d0d7de;padding:6px 8px;text-align:left}}
.good{{color:#1a7f37;font-weight:600}}.bad{{color:#cf222e;font-weight:600}}
tr.critical td:nth-child(2){{color:#cf222e;font-weight:600}}
tr.warning td:nth-child(2){{color:#9a6700;font-weight:600}}
.note{{color:#57606a}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
@media (max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Credit scoring monitoring</h1>
<p>Generated {summary["generated_at"]} &middot; model run
<code>{summary["model"]["mlflow_run_id"]}</code> &middot; baseline AUC
{baseline["roc_auc"]:.3f} &middot; threshold {summary["model"]["threshold"]:.3f}</p>
<h2>Batches</h2>
<table><tr><th>batch</th><th>status</th><th>rows</th><th>drift share</th><th>AUC</th>
<th>critical</th><th>warning</th><th>reports</th><th>scenario</th></tr>{"".join(rows)}</table>
<div class=grid><div>{_fig_html(auc)}</div><div>{_fig_html(rates)}</div></div>
{_fig_html(share)}
{_fig_html(heat)}
<h2>Alerts</h2>
<p class=note>Feature-level warnings are omitted here; see <code>alerts.json</code>.</p>
<table><tr><th>batch</th><th>severity</th><th>alert</th><th>message</th></tr>{alert_rows}</table>
</body></html>"""
    OUT_PATH.write_text(page)
    return str(OUT_PATH)


def main():
    print(build())


if __name__ == "__main__":
    main()
