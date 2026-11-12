import hashlib
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_scoring_drift import config

log = logging.getLogger(__name__)

RENAME = {
    "ID": "id",
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_status_1",
    **{f"PAY_{i}": f"pay_status_{i}" for i in range(2, 7)},
    **{f"BILL_AMT{i}": f"bill_amt_{i}" for i in range(1, 7)},
    **{f"PAY_AMT{i}": f"pay_amt_{i}" for i in range(1, 7)},
    "default.payment.next.month": config.TARGET,
}


def pseudonymize(value, salt: str = config.PSEUDO_SALT) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:16]


def load_raw(path=config.RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path).rename(columns=RENAME)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, config.ID_COL, df.pop("id").map(pseudonymize))

    # undocumented codes in the UCI data dictionary are folded into "others"
    df["education"] = df["education"].where(df["education"].between(1, 4), 4)
    df["marriage"] = df["marriage"].where(df["marriage"].between(1, 3), 3)

    for col in ["limit_bal", *config.BILL_AMT, *config.PAY_AMT]:
        df[col] = df[col].astype(float)

    feature_cols = [c for c in df.columns if c != config.ID_COL]
    before = len(df)
    df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)
    log.info("dropped %d duplicated records", before - len(df))
    return df


def split_reference_production(df: pd.DataFrame, production_share: float = 0.4):
    reference, production = train_test_split(
        df,
        test_size=production_share,
        stratify=df[config.TARGET],
        random_state=config.SEED,
    )
    return reference.reset_index(drop=True), production.reset_index(drop=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = clean(load_raw())
    reference, production = split_reference_production(df)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(config.REFERENCE_PATH, index=False)
    production.to_parquet(config.PRODUCTION_POOL_PATH, index=False)
    log.info("reference: %d rows -> %s", len(reference), config.REFERENCE_PATH)
    log.info("production pool: %d rows -> %s", len(production), config.PRODUCTION_POOL_PATH)


if __name__ == "__main__":
    main()
