"""Bar-close vectorised backtest engine.

Entry at close of bar where signal transitions 0 → ±1.
Exit at close of bar where signal transitions back to 0, or when SL/TP hit intrabar.
Spread deducted on entry (half-spread model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    sl_prices: pd.Series | None = None,
    tp_prices: pd.Series | None = None,
    spread_col: str = "spread",
    initial_capital: float = 10_000.0,
    point_value: float = 1.0,
) -> dict:
    """Run a bar-close backtest.

    Returns dict with keys: equity_curve, trades, metrics.
    """
    df      = df.copy()
    signals = signals.reindex(df.index, fill_value=0)
    spread  = df[spread_col].fillna(0.0) if spread_col in df.columns else pd.Series(0.0, index=df.index)

    pos         = 0
    entry_price = 0.0
    entry_sl    = np.nan
    entry_tp    = np.nan
    entry_time  = None

    trade_records: list[dict] = []
    equity     = initial_capital
    equity_pts: list[float]        = [initial_capital]
    equity_idx: list[pd.Timestamp] = [df.index[0]]

    def _close_trade(exit_price: float, exit_time: pd.Timestamp, reason: str) -> None:
        nonlocal pos, entry_price, entry_sl, entry_tp, entry_time, equity
        pnl = (exit_price - entry_price) * pos * point_value
        r   = pnl / (abs(entry_price - entry_sl) * point_value) if not np.isnan(entry_sl) and entry_sl != entry_price else np.nan
        trade_records.append({
            "entry_time":  entry_time,
            "exit_time":   exit_time,
            "direction":   pos,
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "sl":          entry_sl,
            "tp":          entry_tp,
            "pnl":         pnl,
            "r_multiple":  r,
            "exit_reason": reason,
        })
        equity += pnl
        pos = 0

    prev_sig = 0

    for i, idx in enumerate(df.index):
        row = df.iloc[i]
        sig = int(signals.iloc[i])
        spd = float(spread.iloc[i])

        sl_now = float(sl_prices.iloc[i]) if sl_prices is not None else np.nan
        tp_now = float(tp_prices.iloc[i]) if tp_prices is not None else np.nan

        if pos != 0:
            sl_hit = tp_hit = False

            if pos == 1:
                if not np.isnan(entry_sl) and row["low"] <= entry_sl:
                    sl_hit = True
                if not np.isnan(entry_tp) and row["high"] >= entry_tp:
                    tp_hit = True
            else:
                if not np.isnan(entry_sl) and row["high"] >= entry_sl:
                    sl_hit = True
                if not np.isnan(entry_tp) and row["low"] <= entry_tp:
                    tp_hit = True

            if sl_hit and tp_hit:
                _close_trade(entry_sl, idx, "SL")
            elif sl_hit:
                _close_trade(entry_sl, idx, "SL")
            elif tp_hit:
                _close_trade(entry_tp, idx, "TP")
            elif sig == 0 or (sig != 0 and sig != pos):
                exit_px = row["close"] + (spd / 2.0 if pos == -1 else -spd / 2.0)
                _close_trade(exit_px, idx, "Signal")

        if pos == 0 and sig != 0:
            pos         = sig
            half_spread = spd / 2.0
            entry_price = row["close"] + sig * half_spread
            entry_time  = idx
            entry_sl    = sl_now
            entry_tp    = tp_now

        prev_sig = sig
        equity_pts.append(equity)
        equity_idx.append(idx)

    if pos != 0:
        last_row = df.iloc[-1]
        spd = float(spread.iloc[-1])
        exit_px = last_row["close"] + (spd / 2.0 if pos == -1 else -spd / 2.0)
        _close_trade(exit_px, df.index[-1], "EndOfData")

    equity_curve = pd.Series(equity_pts[1:], index=df.index)
    trades_df    = pd.DataFrame(trade_records)
    metrics      = _compute_metrics(equity_curve, trades_df, initial_capital)

    return {"equity_curve": equity_curve, "trades": trades_df, "metrics": metrics}


def _compute_metrics(equity: pd.Series, trades: pd.DataFrame, initial_capital: float) -> dict:
    if trades.empty:
        return {
            "profit_factor": 0.0, "sharpe": 0.0, "max_dd": 0.0,
            "win_rate": 0.0, "n_trades": 0, "expectancy": 0.0, "avg_r": np.nan,
        }

    pnl    = trades["pnl"]
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.inf
    win_rate      = len(wins) / len(pnl)
    expectancy    = float(pnl.mean())
    avg_r         = float(trades["r_multiple"].mean()) if "r_multiple" in trades.columns else np.nan

    daily_equity  = equity.resample("1D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()
    sharpe = (
        float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
        if daily_returns.std() > 0 else 0.0
    )

    running_max = equity.cummax()
    dd          = (equity - running_max) / running_max.replace(0, np.nan)
    max_dd      = float(dd.min()) * -1

    return {
        "profit_factor": float(profit_factor),
        "sharpe":        float(sharpe),
        "max_dd":        float(max_dd),
        "win_rate":      float(win_rate),
        "n_trades":      int(len(pnl)),
        "expectancy":    float(expectancy),
        "avg_r":         float(avg_r) if not np.isnan(avg_r) else None,
    }


def backtest_summary(result: dict) -> pd.Series:
    """Return key metrics as a named Series."""
    m = result["metrics"]
    return pd.Series({
        "Profit Factor": round(m["profit_factor"], 3),
        "Sharpe":        round(m["sharpe"], 3),
        "Max DD %":      round(m["max_dd"] * 100, 2),
        "Win Rate %":    round(m["win_rate"] * 100, 1),
        "# Trades":      m["n_trades"],
        "Expectancy":    round(m["expectancy"], 4),
        "Avg R":         round(m["avg_r"], 3) if m["avg_r"] is not None else None,
    })
