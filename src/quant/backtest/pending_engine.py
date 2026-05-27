"""Pending-order backtest engine (BuyStop / SellStop with trailing stop).

order_fn signature:
    order_fn(df, bar_idx, has_buy, has_sell, has_open_pos) -> (buy_level | None, sell_level | None)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from quant.backtest.engine import _compute_metrics


class _PendingOrder:
    __slots__ = ("direction", "level", "tp", "sl", "expires_at_idx", "lots")

    def __init__(self, direction: int, level: float, tp: float, sl: float,
                 expires_at_idx: int, lots: float = 1.0) -> None:
        self.direction      = direction
        self.level          = level
        self.tp             = tp
        self.sl             = sl
        self.expires_at_idx = expires_at_idx
        self.lots           = lots


def run_scalper_backtest(
    df: pd.DataFrame,
    order_fn: Callable,
    *,
    tp_points: float,
    sl_points: float,
    tsl_trigger: float,
    tsl_points: float,
    expiration_bars: int,
    order_dist: float,
    point: float = 0.00001,
    spread_col: str = "spread",
    initial_capital: float = 10_000.0,
    point_value: float = 1.0,
) -> dict:
    """Bar-by-bar M1 backtest with pending orders and trailing stop.

    Returns same format as run_backtest(): equity_curve, trades, metrics.
    """
    df     = df.copy()
    spread = df[spread_col].fillna(0.0) if spread_col in df.columns else pd.Series(0.0, index=df.index)

    tp_dist  = tp_points  * point
    sl_dist  = sl_points  * point
    tsl_trig = tsl_trigger * point
    tsl_step = tsl_points  * point
    od_dist  = order_dist  * point

    pending_buy:  _PendingOrder | None = None
    pending_sell: _PendingOrder | None = None

    pos          = 0
    entry_price  = 0.0
    entry_sl     = np.nan
    entry_tp     = np.nan
    entry_time   = None
    entry_level  = np.nan
    entry_bar_hi = np.nan
    trail_sl     = np.nan

    trade_records: list[dict] = []
    equity     = initial_capital
    equity_pts: list[float]        = [initial_capital]
    equity_idx: list[pd.Timestamp] = [df.index[0]]

    def _close_trade(exit_px: float, exit_ts: pd.Timestamp, reason: str) -> None:
        nonlocal pos, entry_price, entry_sl, entry_tp, entry_time, trail_sl, equity
        nonlocal entry_level, entry_bar_hi
        pnl = (exit_px - entry_price) * pos * point_value
        r   = (pnl / (abs(entry_price - entry_sl) * point_value)
               if not np.isnan(entry_sl) and entry_sl != entry_price else np.nan)
        trade_records.append({
            "entry_time": entry_time, "exit_time": exit_ts, "direction": pos,
            "entry_price": entry_price, "exit_price": exit_px,
            "sl": entry_sl, "tp": entry_tp, "pnl": pnl, "r_multiple": r, "exit_reason": reason,
            "pending_level": entry_level, "fill_bar_high": entry_bar_hi,
        })
        equity += pnl
        pos = 0; trail_sl = np.nan

    def _fill_pending(order: _PendingOrder, fill_px: float, ts: pd.Timestamp, bar_high: float) -> None:
        nonlocal pos, entry_price, entry_sl, entry_tp, entry_time, trail_sl, entry_level, entry_bar_hi
        pos = order.direction; entry_price = fill_px; entry_sl = order.sl
        entry_tp = order.tp; entry_time = ts; entry_level = order.level
        entry_bar_hi = bar_high; trail_sl = np.nan

    for i, idx in enumerate(df.index):
        row = df.iloc[i]
        spd = float(spread.iloc[i])
        hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])

        if pos != 0:
            active_sl = trail_sl if not np.isnan(trail_sl) else entry_sl

            if pos == 1:
                unrealised = (hi - entry_price) / point
                if unrealised >= tsl_trigger:
                    new_tsl = hi - tsl_step
                    if np.isnan(trail_sl) or new_tsl > trail_sl:
                        trail_sl = new_tsl; active_sl = trail_sl; entry_sl = active_sl
                sl_hit = (not np.isnan(active_sl)) and lo <= active_sl
                tp_hit = (not np.isnan(entry_tp)) and hi >= entry_tp
            else:
                unrealised = (entry_price - lo) / point
                if unrealised >= tsl_trigger:
                    new_tsl = lo + tsl_step
                    if np.isnan(trail_sl) or new_tsl < trail_sl:
                        trail_sl = new_tsl; active_sl = trail_sl; entry_sl = active_sl
                sl_hit = (not np.isnan(active_sl)) and hi >= active_sl
                tp_hit = (not np.isnan(entry_tp)) and lo <= entry_tp

            if sl_hit and tp_hit:   _close_trade(active_sl, idx, "SL")
            elif sl_hit:            _close_trade(active_sl, idx, "SL")
            elif tp_hit:            _close_trade(entry_tp,  idx, "TP")

        if pos == 0:
            if pending_buy is not None:
                if i >= pending_buy.expires_at_idx:
                    pending_buy = None
                elif hi >= pending_buy.level:
                    _fill_pending(pending_buy, pending_buy.level + spd / 2.0, idx, hi)
                    pending_buy = None

            if pending_sell is not None and pos == 0:
                if i >= pending_sell.expires_at_idx:
                    pending_sell = None
                elif lo <= pending_sell.level:
                    _fill_pending(pending_sell, pending_sell.level - spd / 2.0, idx, lo)
                    pending_sell = None

        has_buy  = pending_buy  is not None
        has_sell = pending_sell is not None
        has_pos  = pos != 0

        buy_level, sell_level = order_fn(df, i, has_buy, has_sell, has_pos)

        if buy_level is not None and not has_buy and not has_pos:
            ask = cl + spd / 2.0
            if ask <= buy_level - od_dist:
                pending_buy = _PendingOrder(1, buy_level, buy_level + tp_dist, buy_level - sl_dist, i + expiration_bars)

        if sell_level is not None and not has_sell and not has_pos:
            bid = cl - spd / 2.0
            if bid >= sell_level + od_dist:
                pending_sell = _PendingOrder(-1, sell_level, sell_level - tp_dist, sell_level + sl_dist, i + expiration_bars)

        equity_pts.append(equity)
        equity_idx.append(idx)

    if pos != 0:
        last = df.iloc[-1]
        spd  = float(spread.iloc[-1])
        exit_px = float(last["close"]) + (spd / 2.0 if pos == -1 else -spd / 2.0)
        _close_trade(exit_px, df.index[-1], "EndOfData")

    equity_curve = pd.Series(equity_pts[1:], index=df.index)
    trades_df    = pd.DataFrame(trade_records)
    return {"equity_curve": equity_curve, "trades": trades_df, "metrics": _compute_metrics(equity_curve, trades_df, initial_capital)}
