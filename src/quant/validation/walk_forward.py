"""Walk-forward validation — percentage-based IS / VAL / OOS split.

Default split: 60% in-sample, 20% validation, 20% out-of-sample.
OOS must never be used during parameter tuning.
Pass: oos_pf / is_pf >= WF_PASS_THRESHOLD (default 0.70).
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

StrategyFn = Callable[[pd.DataFrame, dict[str, Any]], pd.Series]

WF_PASS_THRESHOLD = 0.70


def walk_forward_test(
    strategy_fn: StrategyFn,
    df_ohlcv: pd.DataFrame,
    base_params: dict[str, Any],
    is_pct: float = 0.60,
    val_pct: float = 0.20,
) -> dict:
    """Split df by percentage into IS / VAL / OOS and compare metrics.

    Args:
        strategy_fn: callable(df, params) → signal Series (or tuple (sig, sl, tp))
        df_ohlcv:    full OHLCV DataFrame
        base_params: strategy parameters to use on all windows
        is_pct:      fraction of data for in-sample (default 0.60)
        val_pct:     fraction of data for validation (default 0.20)
    """
    from quant.backtest.engine import run_backtest

    n       = len(df_ohlcv)
    is_end  = int(n * is_pct)
    val_end = int(n * (is_pct + val_pct))

    df_is  = df_ohlcv.iloc[:is_end]
    df_val = df_ohlcv.iloc[is_end:val_end]
    df_oos = df_ohlcv.iloc[val_end:]

    def _metrics(df: pd.DataFrame) -> dict:
        if len(df) < 20:
            return {"pf": float("nan"), "sharpe": float("nan"), "n": 0}
        _raw = strategy_fn(df, base_params)
        if isinstance(_raw, tuple):
            sig = _raw[0]
            sl  = _raw[1] if len(_raw) > 1 else None
            tp  = _raw[2] if len(_raw) > 2 else None
        else:
            sig, sl, tp = _raw, None, None
        result = run_backtest(df, sig, sl_prices=sl, tp_prices=tp)
        m = result["metrics"]
        return {"pf": float(m["profit_factor"]), "sharpe": float(m["sharpe"]), "n": int(m["n_trades"])}

    m_is  = _metrics(df_is)
    m_val = _metrics(df_val)
    m_oos = _metrics(df_oos)

    def _ratio(a: float, b: float) -> float:
        if b in (0.0, float("nan")) or pd.isna(b): return float("nan")
        return float(a / b)

    is_oos_ratio = _ratio(m_oos["pf"], m_is["pf"])
    is_val_ratio = _ratio(m_val["pf"], m_is["pf"])
    passed       = not pd.isna(is_oos_ratio) and is_oos_ratio >= WF_PASS_THRESHOLD

    return {
        "is_pf":        m_is["pf"],   "is_sharpe":  m_is["sharpe"],  "is_n":  m_is["n"],
        "val_pf":       m_val["pf"],  "val_sharpe": m_val["sharpe"], "val_n": m_val["n"],
        "oos_pf":       m_oos["pf"],  "oos_sharpe": m_oos["sharpe"], "oos_n": m_oos["n"],
        "is_oos_ratio": round(is_oos_ratio, 3) if not pd.isna(is_oos_ratio) else float("nan"),
        "is_val_ratio": round(is_val_ratio, 3) if not pd.isna(is_val_ratio) else float("nan"),
        "pass":         bool(passed),
        "split": {
            "is_end_idx":  is_end,
            "val_end_idx": val_end,
            "is_start":    str(df_is.index[0].date())  if len(df_is)  else "",
            "is_end":      str(df_is.index[-1].date()) if len(df_is)  else "",
            "oos_start":   str(df_oos.index[0].date()) if len(df_oos) else "",
        },
    }
