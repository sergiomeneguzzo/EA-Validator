"""Purged k-fold cross-validation for time-series strategies (Lopez de Prado AFML ch.7)."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PurgedKFold:
    """Time-series k-fold splitter that purges the test window from training data."""

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 0.0 <= embargo_pct < 0.5:
            raise ValueError("embargo_pct must be in [0, 0.5)")
        self.n_splits    = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame, y=None, groups=None):
        n          = len(X)
        fold_size  = n // self.n_splits
        embargo_n  = max(1, int(n * self.embargo_pct))

        for i in range(self.n_splits):
            t_start = i * fold_size
            t_end   = (i + 1) * fold_size if i < self.n_splits - 1 else n
            test_idx = np.arange(t_start, t_end)
            embargo_end = min(t_end + embargo_n, n)
            train_idx = np.concatenate([np.arange(0, t_start), np.arange(embargo_end, n)])
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx.astype(int), test_idx.astype(int)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def cross_val_score_purged(
    strategy_fn,
    df_ohlcv: pd.DataFrame,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    sharpe_method: str = "engine",
) -> list[float]:
    """OOS Sharpe ratio per purged fold.

    sharpe_method: 'engine' (realized P&L) or 'signal_returns' (mark-to-market).
    Use 'signal_returns' for long-hold strategies where trades span fold boundaries.
    """
    from quant.backtest.engine import run_backtest

    _raw = strategy_fn(df_ohlcv)
    if isinstance(_raw, tuple):
        signal    = _raw[0]
        sl_series = _raw[1] if len(_raw) > 1 else None
        tp_series = _raw[2] if len(_raw) > 2 else None
    else:
        signal, sl_series, tp_series = _raw, None, None

    pkf = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    oos_sharpes: list[float] = []
    daily_ret_full = df_ohlcv["close"].pct_change()

    for _, test_idx in pkf.split(df_ohlcv):
        df_test  = df_ohlcv.iloc[test_idx]
        sig_test = signal.iloc[test_idx]
        sl_test  = sl_series.iloc[test_idx] if sl_series is not None else None
        tp_test  = tp_series.iloc[test_idx] if tp_series is not None else None

        if sig_test.abs().sum() == 0 or len(df_test) < 20:
            oos_sharpes.append(float("nan")); continue

        if sharpe_method == "signal_returns":
            ret = daily_ret_full.iloc[test_idx]
            pnl = sig_test.shift(1).fillna(0) * ret
            pnl = pnl.dropna()
            sr  = float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 and len(pnl) > 10 else float("nan")
            oos_sharpes.append(sr)
        else:
            result = run_backtest(df_test, sig_test, sl_prices=sl_test, tp_prices=tp_test)
            oos_sharpes.append(result["metrics"]["sharpe"])

    return oos_sharpes
