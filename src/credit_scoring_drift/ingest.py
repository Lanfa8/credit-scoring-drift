import argparse
import json
import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from credit_scoring_drift import config, contracts

log = logging.getLogger(__name__)

QUALITY_DIR = config.REPORTS_DIR / "quality"


class IngestionBlocked(Exception):
    def __init__(self, batch_id: str, result: contracts.ValidationResult):
        self.batch_id = batch_id
        self.result = result
        super().__init__(
            f"batch {batch_id} rejected by data contract: "
            f"{result.failed_rows} invalid rows, {len(result.failure_cases)} violations"
        )


def _write_quality_report(batch_id: str, result: contracts.ValidationResult) -> Path:
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch_id": batch_id,
        "validated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "contract": contracts.credit_contract.name,
        "passed": result.passed,
        "rows": result.n_rows,
        "invalid_rows": result.failed_rows,
        "violations_by_check": result.summary_by_check(),
    }
    (QUALITY_DIR / f"{batch_id}.json").write_text(json.dumps(payload, indent=2))

    status = "APPROVED" if result.passed else "BLOCKED"
    color = "#1a7f37" if result.passed else "#cf222e"
    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in payload["violations_by_check"].items()
    )
    sample = ""
    if not result.passed:
        cols = ["column", "check", "failure_case", "index"]
        sample = result.failure_cases[cols].head(50).to_html(index=False, border=0)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Data contract - {batch_id}</title>
<style>body{{font-family:sans-serif;margin:2rem;max-width:960px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:4px 8px}}</style></head>
<body><h1>Data contract: {batch_id}</h1>
<p>Contract <code>{payload["contract"]}</code> &middot; {payload["validated_at"]}</p>
<h2 style="color:{color}">{status}</h2>
<p>{payload["rows"]} rows, {payload["invalid_rows"]} invalid rows</p>
<h3>Violations by check</h3><table><tr><th>column:check</th><th>count</th></tr>{rows}</table>
<h3>Failure cases (first 50)</h3>{sample}</body></html>"""
    path = QUALITY_DIR / f"{batch_id}.html"
    path.write_text(html)
    return path


def ingest(path: Path, batch_id: str | None = None) -> pd.DataFrame:
    path = Path(path)
    batch_id = batch_id or path.stem
    df = pd.read_csv(path, dtype={config.ID_COL: str})
    result = contracts.validate(df)
    report = _write_quality_report(batch_id, result)

    if not result.passed:
        target = config.QUARANTINE_DIR / batch_id
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target / path.name)
        result.failure_cases.to_csv(target / "failure_cases.csv", index=False)
        log.error(
            "batch blocked by data contract",
            extra={
                "batch_id": batch_id,
                "invalid_rows": result.failed_rows,
                "violations": result.summary_by_check(),
                "quarantine": str(target),
                "report": str(report),
            },
        )
        raise IngestionBlocked(batch_id, result)

    log.info("batch accepted", extra={"batch_id": batch_id, "rows": result.n_rows})
    return result.data


def main():
    parser = argparse.ArgumentParser(description="Validate a batch against the data contract")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        df = ingest(args.path)
    except IngestionBlocked as exc:
        print(exc)
        for check, count in exc.result.summary_by_check().items():
            print(f"  - {check}: {count}")
        sys.exit(1)
    print(f"batch accepted: {len(df)} rows")


if __name__ == "__main__":
    main()
