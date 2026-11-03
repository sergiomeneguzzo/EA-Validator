import math

import numpy as np
import pandas as pd
import pytest

from quant.validation.deflated_sharpe import annualise_sharpe, deflated_sharpe, sharpe_from_returns
from quant.validation.monte_carlo import mc_bootstrap_test, mc_permutation_test


def test_sharpe_from_returns_is_per_period():
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 1000))
    sr, _, _, n = sharpe_from_returns(r)
    assert n == 1000
    assert sr == pytest.approx(r.mean() / r.std(ddof=1))
    assert annualise_sharpe(sr) == pytest.approx(sr * math.sqrt(252))


def test_weak_strategy_is_not_significant_after_many_trials():
    r = pd.Series(np.random.default_rng(1).normal(0.0002, 0.01, 500))
    sr, skew, kurt, n = sharpe_from_returns(r)
    _, p_value = deflated_sharpe(sr, n_trials=50, skewness=skew, excess_kurtosis=kurt, n_obs=n,
                                 sr_std=0.30 / math.sqrt(252))
    assert p_value > 0.05


def test_more_trials_lower_the_deflated_sharpe():
    r = pd.Series(np.random.default_rng(2).normal(0.002, 0.01, 750))
    sr, skew, kurt, n = sharpe_from_returns(r)
    few, _ = deflated_sharpe(sr, 2, skew, kurt, n, sr_std=0.3 / math.sqrt(252))
    many, _ = deflated_sharpe(sr, 500, skew, kurt, n, sr_std=0.3 / math.sqrt(252))
    assert few > many


def test_bootstrap_varies_the_final_equity():
    trades = np.random.default_rng(3).normal(0.1, 1.0, 300)
    boot = mc_bootstrap_test(trades, n_simulations=2000)
    assert boot.final_equity_p5 < boot.observed_final < boot.final_equity_p95
    assert 0.0 < boot.p_profitable < 1.0


def test_permutation_keeps_the_final_equity():
    trades = np.random.default_rng(4).normal(0.1, 1.0, 300)
    perm = mc_permutation_test(trades, n_simulations=500)
    assert np.allclose(perm.all_finals, trades.sum())
    assert perm.p_profitable in (0.0, 1.0)
