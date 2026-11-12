import numpy as np
import pandas as pd

from credit_scoring_drift import config

DERIVED = config.DERIVED_FEATURES


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[config.MODEL_FEATURES].copy()
    out["utilization"] = out["bill_amt_1"] / out["limit_bal"].clip(lower=1)
    out["payment_ratio"] = out["pay_amt_1"] / out["bill_amt_2"].clip(lower=1)
    out["payment_ratio"] = out["payment_ratio"].clip(upper=10)
    out["months_delayed"] = (out[config.PAY_STATUS] > 0).sum(axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[0, 25, 40, 60, 200], labels=["18-25", "26-40", "41-60", "60+"])
