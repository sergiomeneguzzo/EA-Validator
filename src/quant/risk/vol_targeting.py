"""Volatility targeting — scale position sizes to hit a target annual vol."""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SCALE = 0.1
MAX_SCALE = 3.0

TARGET_VOL_STRATEGY  = 0.05   # 5%  per-strategy
TARGET_VOL_PORTFOLIO = 0.10   # 10% portfolio


class VolTargeter:
    """Multiplicative scaling factor to hit `target_vol_annual`."""

    def __init__(
        self,
        target_vol_annual: float,
        lookback_days: int = 60,
        min_scale: float = MIN_SCALE,
        max_scale: float = MAX_SCALE,
    ) -> None:
        if target_vol_annual <= 0:
            raise ValueError("target_vol_annual must be positive")
        if lookback_days < 2:
            raise ValueError("lookback_days must be >= 2")
        self.target_vol = target_vol_annual
        self.lookback   = lookback_days
        self.min_scale  = min_scale
        self.max_scale  = max_scale

    def compute_scaling(self, returns: pd.Series) -> float:
        """Current scale based on the most recent `lookback_days`. Returns float in [min, max]."""
        window   = returns.iloc[-self.lookback:]
        realised = self._annualised_vol(window)
        if realised == 0.0:
            return self.max_scale
        return float(np.clip(self.target_vol / realised, self.min_scale, self.max_scale))

    def compute_scaling_series(self, returns: pd.Series) -> pd.Series:
        """Rolling scaling factor aligned to `returns.index`. First lookback-1 values are NaN."""
        def _scale(window: pd.Series) -> float:
            rv = self._annualised_vol(window)
            return float(np.clip(self.target_vol / rv, self.min_scale, self.max_scale)) if rv > 0 else self.max_scale
        return returns.rolling(self.lookback).apply(_scale, raw=False)

    def _annualised_vol(self, returns: pd.Series) -> float:
        return float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) >= 2 else 0.0
