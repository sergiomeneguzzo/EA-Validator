"""Tick-level backtest engine.

Uses actual bid/ask prices for fills and SL detection instead of bar High/Low.
Aggregates ticks to M1 internally, then replays each trade tick-by-tick.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytz

from quant.backtest.engine import _compute_metrics

BERLIN = pytz.timezone("Europe/Berlin")


def _agg_ticks_to_m1(tick_df: pd.DataFrame) -> pd.DataFrame:
    df = tick_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("tick_df must have a DatetimeIndex.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(BERLIN)
    elif str(df.index.tz) != "Europe/Berlin":
        df.index = df.index.tz_convert(BERLIN)

    df["spread_val"] = (df["ask"] - df["bid"]).abs()
    m1 = df.resample("1min").agg(
        open=("bid", "first"), high=("bid", "max"),
        low=("bid", "min"),   close=("bid", "last"),
        volume=("bid", "count"), spread=("spread_val", "mean"),
    )
    return m1.dropna(subset=["open"])


def _ensure_berlin(tick_df: pd.DataFrame) -> pd.DataFrame:
    df = tick_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("tick_df must have a DatetimeIndex.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(BERLIN)
    elif str(df.index.tz) != "Europe/Berlin":
        df.index = df.index.tz_convert(BERLIN)
    return df


class TickBacktestEngine:
    """Tick-level backtest — bid/ask fills, tick-accurate SL detection."""

    def run(self, tick_df: pd.DataFrame, strategy: Any, params: dict | None = None) -> dict:
        """Run tick-level backtest.

        tick_df must have bid and ask columns with a DatetimeIndex.
        strategy must implement generate_signal(df_m1) -> (signals, sl_prices).
        Returns same format as run_backtest() for direct comparison.
        """
        params = params or {}
        initial_capital = float(params.get("initial_capital", 10_000.0))
        point_value     = float(params.get("point_value", 1.0))

        tick_df = _ensure_berlin(tick_df)
        if "bid" not in tick_df.columns or "ask" not in tick_df.columns:
            raise ValueError("tick_df must have 'bid' and 'ask' columns.")

        df_m1 = _agg_ticks_to_m1(tick_df)
        signals, sl_prices = strategy.generate_signal(df_m1)

        tick_df = tick_df.copy()
        tick_df["_min_bucket"] = tick_df.index.floor("1min")

        trade_records: list[dict] = []
        equity     = initial_capital
        equity_pts: list[float]        = [initial_capital]
        equity_idx: list[pd.Timestamp] = [df_m1.index[0]]

        pos         = 0
        entry_price = 0.0
        entry_sl    = np.nan
        entry_tp    = np.nan
        entry_time  = None
        prev_sig    = 0

        def _close_trade(exit_px: float, exit_ts: pd.Timestamp, reason: str) -> None:
            nonlocal pos, entry_price, entry_sl, entry_tp, entry_time, equity
            pnl = (exit_px - entry_price) * pos * point_value
            r   = (pnl / (abs(entry_price - entry_sl) * point_value)
                   if not np.isnan(entry_sl) and entry_sl != entry_price else np.nan)
            trade_records.append({
                "entry_time": entry_time, "exit_time": exit_ts,
                "direction": pos, "entry_price": entry_price, "exit_price": exit_px,
                "sl": entry_sl, "tp": entry_tp, "pnl": pnl, "r_multiple": r, "exit_reason": reason,
            })
            equity += pnl
            pos = 0

        for i, bar_ts in enumerate(df_m1.index):
            sig    = int(signals.iloc[i])
            sl_now = float(sl_prices.iloc[i]) if sl_prices is not None else np.nan
            bar_ticks = tick_df[tick_df["_min_bucket"] == bar_ts]

            if pos != 0 and not bar_ticks.empty:
                exit_done = False
                for tick_ts, tick in bar_ticks.iterrows():
                    bid_px = float(tick["bid"])
                    ask_px = float(tick["ask"])

                    if pos == 1 and not np.isnan(entry_sl) and bid_px <= entry_sl:
                        _close_trade(entry_sl, tick_ts, "SL"); exit_done = True; break
                    if pos == -1 and not np.isnan(entry_sl) and ask_px >= entry_sl:
                        _close_trade(entry_sl, tick_ts, "SL"); exit_done = True; break
                    if pos == 1 and not np.isnan(entry_tp) and ask_px >= entry_tp:
                        _close_trade(entry_tp, tick_ts, "TP"); exit_done = True; break
                    if pos == -1 and not np.isnan(entry_tp) and bid_px <= entry_tp:
                        _close_trade(entry_tp, tick_ts, "TP"); exit_done = True; break

                if not exit_done and pos != 0 and (sig == 0 or sig != pos):
                    if not bar_ticks.empty:
                        tick    = bar_ticks.iloc[0]
                        tick_ts = bar_ticks.index[0]
                        exit_px = float(tick["bid"]) if pos == 1 else float(tick["ask"])
                        _close_trade(exit_px, tick_ts, "Signal")

            if pos == 0 and sig != 0 and prev_sig == 0:
                prev_close = float(df_m1.iloc[i - 1]["close"]) if i > 0 else float(df_m1.iloc[i]["open"])
                filled = False
                for tick_ts, tick in bar_ticks.iterrows():
                    bid_px = float(tick["bid"])
                    ask_px = float(tick["ask"])
                    if sig == 1 and ask_px > prev_close:
                        pos, entry_price, entry_time, entry_sl = 1, ask_px, tick_ts, sl_now
                        filled = True; break
                    if sig == -1 and bid_px < prev_close:
                        pos, entry_price, entry_time, entry_sl = -1, bid_px, tick_ts, sl_now
                        filled = True; break

                if not filled and not bar_ticks.empty:
                    tick    = bar_ticks.iloc[-1]
                    tick_ts = bar_ticks.index[-1]
                    pos         = sig
                    entry_price = float(tick["ask"]) if sig == 1 else float(tick["bid"])
                    entry_time  = tick_ts
                    entry_sl    = sl_now

            prev_sig = sig
            equity_pts.append(equity)
            equity_idx.append(bar_ts)

        if pos != 0 and not tick_df.empty:
            last_tick = tick_df.iloc[-1]
            exit_px   = float(last_tick["bid"]) if pos == 1 else float(last_tick["ask"])
            _close_trade(exit_px, tick_df.index[-1], "EndOfData")

        equity_curve = pd.Series(equity_pts[1:], index=df_m1.index)
        trades_df    = pd.DataFrame(trade_records)

        return {
            "equity_curve": equity_curve,
            "trades":       trades_df,
            "metrics":      _compute_metrics(equity_curve, trades_df, initial_capital),
            "df_m1":        df_m1,
        }


def compare_bar_vs_tick(bar_result: dict, tick_result: dict, symbol: str = "") -> pd.DataFrame:
    """Side-by-side metrics comparison: Bar vs Tick engine."""
    bm = bar_result["metrics"]
    tm = tick_result["metrics"]
    keys   = ["profit_factor", "sharpe", "max_dd", "win_rate", "n_trades", "expectancy"]
    labels = ["Profit Factor", "Sharpe", "Max DD", "Win Rate", "# Trades", "Expectancy"]
    rows = [
        {"Metric": lbl, "Bar": bm.get(k), "Tick": tm.get(k),
         "Delta": (tm.get(k) - bm.get(k)) if bm.get(k) is not None and tm.get(k) is not None else None}
        for k, lbl in zip(keys, labels)
    ]
    df = pd.DataFrame(rows).set_index("Metric")
    if symbol:
        df.index.name = f"Metric ({symbol})"
    return df


def load_tick_sample(
    csv_path: str,
    start: str | None = None,
    end: str | None = None,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Load a date-filtered slice of a raw tick CSV (MT5 format: YYYY.MM.DD HH:MM:SS.mmm,bid,ask)."""
    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts   = pd.Timestamp(end,   tz="UTC") if end   else None
    chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(csv_path, header=None, names=["dt", "bid", "ask"],
                         chunksize=chunksize, on_bad_lines="skip")

    for chunk in reader:
        chunk["dt"] = pd.to_datetime(chunk["dt"], format="%Y.%m.%d %H:%M:%S.%f", errors="coerce")
        chunk = chunk.dropna(subset=["dt"])
        chunk["dt"] = chunk["dt"].dt.tz_localize("UTC").dt.tz_convert(BERLIN)

        if start_ts is not None:
            chunk = chunk[chunk["dt"] >= start_ts.tz_convert(BERLIN)]
        if end_ts is not None:
            chunk = chunk[chunk["dt"] <= end_ts.tz_convert(BERLIN)]

        if not chunk.empty:
            chunks.append(chunk.set_index("dt")[["bid", "ask"]])

    if not chunks:
        warnings.warn(f"No ticks found for period {start} – {end}.", stacklevel=2)
        return pd.DataFrame(columns=["bid", "ask"])

    return pd.concat(chunks).sort_index()
