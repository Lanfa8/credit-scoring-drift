import numpy as np
import pandas as pd
import pytest

from credit_scoring_drift import config, drift


@pytest.fixture
def rng():
    return np.random.default_rng(1)


def test_psi_is_near_zero_for_same_distribution(rng):
    a = pd.Series(rng.normal(0, 1, 5000))
    b = pd.Series(rng.normal(0, 1, 5000))
    assert drift.psi(a, b) < 0.01


def test_psi_detects_mean_shift(rng):
    a = pd.Series(rng.normal(0, 1, 5000))
    b = pd.Series(rng.normal(1, 1, 5000))
    assert drift.psi(a, b) > drift.PSI_SEVERE


def test_psi_handles_categorical_with_new_category():
    a = pd.Series([0, 1, 2] * 100)
    b = pd.Series([0, 1, 2, 3] * 75)
    assert drift.psi(a, b) > 0


def test_statistical_tests_flag_only_shifted_column(valid_batch, rng):
    reference = pd.concat([valid_batch] * 10, ignore_index=True)
    current = reference.sample(1000, random_state=0).copy()
    current["age"] = current["age"] - 15
    stats = drift.statistical_tests(reference, current, config.MODEL_FEATURES)
    flagged = stats.loc[stats["drift"], "feature"].tolist()
    assert flagged == ["age"]
