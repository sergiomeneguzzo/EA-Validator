"""ADX-based market regime detector — exact replica of RegimeDetector.mqh (MQL5).

States: TREND_UP / TREND_DOWN / RANGING / VOLATILE
All maths use Wilder's Smoothed MA (MODE_SMMA = EWM alpha=1/period, adjust=False).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TREND_UP   = 0
TREND_DOWN = 1
RANGING    = 2
VOLATILE   = 3

REGIME_NAMES = {TREND_UP: "TREND_UP", TREND_DOWN: "TREND_DOWN", RANGING: "RANGING", VOLATILE: "VOLATILE"}


def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    values = series.to_numpy(dtype=float, na_value=np.nan)
    result = np.full(len(values), np.nan)
    start  = 0
    while start < len(values) and np.isnan(values[start]):
        start += 1
    first_valid = start + period - 1
    if first_valid >= len(values):
        return pd.Series(result, index=series.index)
    result[first_valid] = np.nanmean(values[start : start + period])
    k = 1.0 / period
    one_minus_k = 1.0 - k
    for i in range(first_valid + 1, len(values)):
        result[i] = result[i - 1] if np.isnan(values[i]) else result[i - 1] * one_minus_k + values[i] * k
    return pd.Series(result, index=series.index)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR — identical to MQL5 iATR."""
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    tr.iloc[0] = h.iloc[0] - l.iloc[0]
    return _wilder_rma(tr, period)


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX, +DI, -DI — identical to MQL5 iADX. Returns DataFrame with adx, di_plus, di_minus, atr."""
    h, l, c = df["high"], df["low"], df["close"]
    prev_h, prev_l, prev_c = h.shift(1), l.shift(1), c.shift(1)

    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    tr.iloc[0] = h.iloc[0] - l.iloc[0]

    up, down = h - prev_h, prev_l - l
    dm_plus  = pd.Series(np.where((up > down) & (up > 0),   up,   0.0), index=df.index)
    dm_minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    dm_plus.iloc[0] = dm_minus.iloc[0] = 0.0

    atr14   = _wilder_rma(tr, period)
    di_plus  = 100.0 * _wilder_rma(dm_plus,  period) / atr14.replace(0, np.nan)
    di_minus = 100.0 * _wilder_rma(dm_minus, period) / atr14.replace(0, np.nan)
    di_sum   = di_plus + di_minus
    dx  = (100.0 * (di_plus - di_minus).abs() / di_sum.replace(0, np.nan)).fillna(0.0)
    adx = _wilder_rma(dx, period)

    return pd.DataFrame({"adx": adx, "di_plus": di_plus, "di_minus": di_minus, "atr": atr14}, index=df.index)


class AdxRegimeDetector:
    """Vectorised replica of CRegimeDetector from RegimeDetector.mqh."""

    def __init__(
        self,
        adx_period: int = 14,
        atr_period: int = 14,
        atr_ma_period: int = 20,
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 20.0,
        atr_volatile_multiplier: float = 1.5,
    ) -> None:
        self.adx_period              = adx_period
        self.atr_period              = atr_period
        self.atr_ma_period           = atr_ma_period
        self.adx_trend_threshold     = adx_trend_threshold
        self.adx_range_threshold     = adx_range_threshold
        self.atr_volatile_multiplier = atr_volatile_multiplier

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Return pd.Series[int] with TREND_UP/TREND_DOWN/RANGING/VOLATILE per bar."""
        atr       = compute_atr(df, self.atr_period)
        atr_mean  = atr.rolling(self.atr_ma_period).mean()
        volatile  = atr > atr_mean * self.atr_volatile_multiplier

        adx_df   = compute_adx(df, self.adx_period)
        adx      = adx_df["adx"]
        di_plus  = adx_df["di_plus"]
        di_minus = adx_df["di_minus"]

        regime = pd.Series(RANGING, index=df.index, dtype=int)
        non_volatile = ~volatile
        trending     = adx > self.adx_trend_threshold

        regime[non_volatile & trending & (di_plus >  di_minus)] = TREND_UP
        regime[non_volatile & trending & (di_plus <= di_minus)] = TREND_DOWN
        regime[volatile] = VOLATILE

        return regime

    def detect_named(self, df: pd.DataFrame) -> pd.Series:
        return self.detect(df).map(REGIME_NAMES)
