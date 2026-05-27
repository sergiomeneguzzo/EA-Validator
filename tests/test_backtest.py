"""Smoke tests for the backtest engines."""

import numpy as np
import pandas as pd

from quant.backtest.engine import run_backtest
from quant.validation.scorecard import StrategyScorecard, ScorecardConfig


def _sample_df(n: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    c = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c}, index=idx)


def _trending_signal(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1, index=df.index)


class TestBarCloseEngine:
    def test_returns_required_keys(self):
        df = _sample_df()
        sig = _trending_signal(df)
        result = run_backtest(df, sig)
        assert {"equity_curve", "trades", "metrics"}.issubset(result.keys())

    def test_equity_curve_starts_at_capital(self):
        df = _sample_df()
        sig = _trending_signal(df)
        result = run_backtest(df, sig, initial_capital=10_000.0)
        assert result["equity_curve"].iloc[0] == pytest.approx(10_000.0, rel=0.01)

    def test_metrics_types(self):
        df = _sample_df()
        result = run_backtest(df, _trending_signal(df))
        m = result["metrics"]
        assert isinstance(m["profit_factor"], float)
        assert isinstance(m["sharpe"], float)
        assert 0.0 <= m["win_rate"] <= 1.0


import pytest


class TestScorecard:
    def test_evaluate_returns_verdict(self):
        df = _sample_df()
        sig = _trending_signal(df)
        result = run_backtest(df, sig)
        card = StrategyScorecard(config=ScorecardConfig()).evaluate(
            df=df, signal=sig, backtest_result=result
        )
        assert card["verdict"] in {"green", "yellow", "red"}
        assert "scores" in card
