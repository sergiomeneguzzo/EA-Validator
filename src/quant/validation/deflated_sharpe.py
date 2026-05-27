"""Deflated Sharpe Ratio — Bailey & Lopez de Prado (2014), SSRN 2460551."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

_EULER_GAMMA = 0.5772156649


def probabilistic_sharpe_ratio(
    sr_hat: float,
    sr_benchmark: float = 0.0,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    n_obs: int = 252,
) -> float:
    """P(SR_true > sr_benchmark) under the Bailey-LdP asymptotic formula."""
    if n_obs < 2:
        return float("nan")
    variance = _sr_variance(sr_hat, skewness, excess_kurtosis, n_obs)
    if variance <= 0:
        return 1.0 if sr_hat > sr_benchmark else 0.0
    z = (sr_hat - sr_benchmark) / math.sqrt(variance)
    return float(norm.cdf(z))


def deflated_sharpe(
    sr_hat: float,
    n_trials: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    n_obs: int = 252,
    sr_mean: float = 0.0,
    sr_std: float = 1.0,
) -> tuple[float, float]:
    """DSR corrected for selection bias across n_trials. Returns (dsr, p_value)."""
    sr_star = _expected_max_sr(n_trials, sr_mean, sr_std)
    dsr     = probabilistic_sharpe_ratio(sr_hat, sr_benchmark=sr_star,
                                          skewness=skewness, excess_kurtosis=excess_kurtosis, n_obs=n_obs)
    return float(dsr), float(1.0 - dsr)


def sharpe_from_returns(returns: "pd.Series") -> tuple[float, float, float, int]:
    """Compute annualised SR, skewness, excess_kurtosis, n_obs from a return series."""
    import pandas as pd
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 4 or r.std(ddof=1) == 0:
        return 0.0, 0.0, 0.0, n
    mu  = float(r.mean())
    sig = float(r.std(ddof=1))
    sr  = mu / sig * math.sqrt(252)
    z   = (r - mu) / sig
    sk  = float((z ** 3).mean())
    ku  = float((z ** 4).mean()) - 3.0
    return sr, sk, ku, n


def _sr_variance(sr: float, skew: float, exc_kurt: float, n: int) -> float:
    var = (1.0 - skew * sr + (exc_kurt + 2.0) / 4.0 * sr ** 2) / max(n - 1, 1)
    return max(var, 1e-12)


def _expected_max_sr(n_trials: int, mean: float = 0.0, std: float = 1.0) -> float:
    if n_trials <= 1:
        return mean
    n     = max(n_trials, 2)
    gamma = _EULER_GAMMA
    q1    = min(1.0 - 1.0 / n,           1.0 - 1e-10)
    q2    = min(1.0 - 1.0 / (n * math.e), 1.0 - 1e-10)
    emax  = (1.0 - gamma) * norm.ppf(q1) + gamma * norm.ppf(q2)
    return float(mean + std * emax)
