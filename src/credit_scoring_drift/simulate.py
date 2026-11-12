import json
import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from credit_scoring_drift import config

log = logging.getLogger(__name__)

BATCH_SIZE = 2000


@dataclass
class Scenario:
    period: str
    intensity: float
    concept_drift: bool
    notes: str


SCENARIOS = [
    Scenario("2025-01", 0.0, False, "stable period, same profile as reference"),
    Scenario("2025-02", 0.1, False, "early inflation signals"),
    Scenario("2025-03", 0.3, False, "inflation rising, younger customers arriving"),
    Scenario("2025-04", 0.5, True, "income squeeze, leveraged customers start defaulting"),
    Scenario("2025-05", 0.75, True, "unemployment up, strong profile change"),
    Scenario("2025-06", 1.0, True, "full economic shock"),
]


def sample_population(pool: pd.DataFrame, t: float, rng: np.random.Generator) -> pd.DataFrame:
    """Covariate shift: oversample younger and already-delinquent customers.

    Resampling keeps P(default | x) intact, so this part is pure data drift.
    """
    delayed = (pool["pay_status_1"] >= 1).astype(float)
    weights = np.exp(1.5 * t * delayed) * np.exp(-1.2 * t * (pool["age"] - 35) / 10)
    weights = weights / weights.sum()
    idx = rng.choice(len(pool), size=BATCH_SIZE, replace=False, p=weights)
    return pool.iloc[idx].reset_index(drop=True)


def apply_economic_shock(df: pd.DataFrame, t: float, rng: np.random.Generator) -> pd.DataFrame:
    """Nominal values inflate; a growing share of customers can only pay half of what
    they used to pay, while credit limits stay frozen."""
    df = df.copy()
    inflation = (1 + 0.6 * t) * rng.lognormal(0, 0.03 * t, size=(len(df), 1))
    squeezed = rng.random(len(df)) < 0.5 * t
    pay_factor = inflation * np.where(squeezed, 0.5, 1.0)[:, None]
    df[config.BILL_AMT] = (df[config.BILL_AMT].to_numpy() * inflation).round()
    df[config.PAY_AMT] = (df[config.PAY_AMT].to_numpy() * pay_factor).round()
    return df


def apply_concept_drift(df: pd.DataFrame, t: float, rng: np.random.Generator) -> pd.DataFrame:
    """Same features, different outcome: customers that always paid on time but are
    highly leveraged start defaulting, a pattern the reference data never showed."""
    df = df.copy()
    utilization = df["bill_amt_1"] / df["limit_bal"]
    flip_prob = 0.6 * max(t - 0.4, 0) / 0.6
    on_time = df["pay_status_1"] <= 0
    at_risk = (df[config.TARGET] == 0) & on_time & (utilization > 0.5)
    flips = at_risk & (rng.random(len(df)) < flip_prob)
    df.loc[flips, config.TARGET] = 1
    return df


def corrupt(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Upstream bug: broken extraction job delivering a dirty batch."""
    df = df.copy()
    n = len(df)
    pick = lambda k: rng.choice(n, size=k, replace=False)  # noqa: E731
    df.loc[pick(60), "limit_bal"] = np.nan
    df.loc[pick(25), "age"] = rng.choice([0, 15, 17], size=25)
    df.loc[pick(15), "education"] = 0
    df.loc[pick(30), "pay_amt_1"] *= -1
    df["cpf"] = [f"{rng.integers(10**10, 10**11)}" for _ in range(n)]
    return pd.concat([df, df.sample(25, random_state=1)], ignore_index=True)


def simulate(seed: int = config.SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    pool = pd.read_parquet(config.PRODUCTION_POOL_PATH)
    config.BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for sc in SCENARIOS:
        batch = sample_population(pool, sc.intensity, rng)
        batch = apply_economic_shock(batch, sc.intensity, rng)
        default_before = batch[config.TARGET].mean()
        if sc.concept_drift:
            batch = apply_concept_drift(batch, sc.intensity, rng)
        injected = round(float(batch[config.TARGET].mean() - default_before), 4)

        if sc.period == "2025-03":
            bad_id = f"{sc.period}-upstream-bug"
            corrupt(batch, rng).to_csv(config.BATCHES_DIR / f"{bad_id}.csv", index=False)
            manifest.append(
                {
                    "batch_id": bad_id,
                    "file": f"{bad_id}.csv",
                    **asdict(sc),
                    "notes": "corrupted extraction: nulls, invalid ages, legacy codes, "
                    "negative payments, duplicates and a leaked CPF column",
                }
            )

        batch.to_csv(config.BATCHES_DIR / f"{sc.period}.csv", index=False)
        manifest.append(
            {
                "batch_id": sc.period,
                "file": f"{sc.period}.csv",
                **asdict(sc),
                "injected_concept_default": injected,
            }
        )
        log.info(
            "%s: intensity=%.2f concept=%s default_rate=%.3f",
            sc.period,
            sc.intensity,
            sc.concept_drift,
            batch[config.TARGET].mean(),
        )

    (config.BATCHES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    simulate()


if __name__ == "__main__":
    main()
