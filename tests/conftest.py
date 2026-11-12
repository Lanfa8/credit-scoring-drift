import numpy as np
import pandas as pd
import pytest

from credit_scoring_drift import config
from credit_scoring_drift.data import pseudonymize


@pytest.fixture
def valid_batch() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            config.ID_COL: [pseudonymize(i) for i in range(n)],
            "limit_bal": rng.integers(10, 500, n) * 1000.0,
            "sex": rng.integers(1, 3, n),
            "education": rng.integers(1, 5, n),
            "marriage": rng.integers(1, 4, n),
            "age": rng.integers(21, 70, n),
        }
    )
    for col in config.PAY_STATUS:
        df[col] = rng.integers(-2, 4, n)
    for col in config.BILL_AMT:
        df[col] = rng.normal(40_000, 20_000, n).round()
    for col in config.PAY_AMT:
        df[col] = rng.exponential(3_000, n).round()
    df[config.TARGET] = rng.integers(0, 2, n)
    return df
