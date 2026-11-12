import pandas as pd
import pytest

from credit_scoring_drift import config, contracts
from credit_scoring_drift.ingest import IngestionBlocked, ingest


def test_valid_batch_passes(valid_batch):
    assert contracts.validate(valid_batch).passed


@pytest.mark.parametrize(
    "column,value,check",
    [
        ("age", 16, "age:in_range(18, 100)"),
        ("limit_bal", None, "limit_bal:not_nullable"),
        ("limit_bal", -5_000.0, "limit_bal:in_range(1, 1000000)"),
        ("education", 7, "education:isin([1, 2, 3, 4])"),
        ("pay_amt_1", -10.0, "pay_amt_1:in_range(0, 3000000)"),
    ],
)
def test_rule_violations_are_caught(valid_batch, column, value, check):
    valid_batch.loc[3, column] = value
    result = contracts.validate(valid_batch)
    assert not result.passed
    assert check in result.summary_by_check()


def test_duplicates_are_rejected(valid_batch):
    batch = pd.concat([valid_batch, valid_batch.iloc[[0]]], ignore_index=True)
    result = contracts.validate(batch)
    assert not result.passed
    checks = result.summary_by_check()
    assert f"{config.ID_COL}:field_uniqueness" in checks
    assert any("no_duplicated_records" in k for k in checks)


def test_unexpected_column_is_rejected(valid_batch):
    valid_batch["cpf"] = "000.000.000-00"
    assert not contracts.validate(valid_batch).passed


def test_small_batch_is_rejected(valid_batch):
    assert not contracts.validate(valid_batch.head(10)).passed


def test_ingest_blocks_and_quarantines(valid_batch, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "QUARANTINE_DIR", tmp_path / "quarantine")
    monkeypatch.setattr("credit_scoring_drift.ingest.QUALITY_DIR", tmp_path / "quality")
    valid_batch.loc[0, "age"] = 12
    path = tmp_path / "bad.csv"
    valid_batch.to_csv(path, index=False)

    with pytest.raises(IngestionBlocked):
        ingest(path)
    assert (tmp_path / "quarantine" / "bad" / "failure_cases.csv").exists()
    assert (tmp_path / "quality" / "bad.json").exists()
