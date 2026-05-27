"""Drawdown controller — reduce position size when equity is in drawdown.

thresholds: list of (upper_dd_bound, scale) sorted ascending.
Example: [(0.03, 1.0), (0.06, 0.5), (0.10, 0.25), (1.0, 0.0)]
  current_dd=0.02 → scale=1.0, current_dd=0.07 → scale=0.25
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = [
    (0.03, 1.00),
    (0.06, 0.50),
    (0.10, 0.25),
    (1.00, 0.00),
]


class DrawdownController:
    """Piecewise scaling based on equity drawdown from high-water mark."""

    def __init__(self, thresholds: list[tuple[float, float]] = DEFAULT_THRESHOLDS) -> None:
        if not thresholds:
            raise ValueError("thresholds must not be empty")
        self.thresholds = sorted(thresholds, key=lambda x: x[0])

    def compute_scaling(self, equity_curve: pd.Series) -> float:
        return self._dd_to_scale(self._current_dd(equity_curve))

    def compute_scaling_series(self, equity_curve: pd.Series) -> pd.Series:
        hwm      = equity_curve.cummax()
        safe_hwm = hwm.replace(0, np.nan).ffill().fillna(1.0)
        dd_series = ((equity_curve - hwm) / safe_hwm).clip(upper=0).abs()
        return dd_series.apply(self._dd_to_scale)

    def current_dd(self, equity_curve: pd.Series) -> float:
        return self._current_dd(equity_curve)

    def _current_dd(self, equity_curve: pd.Series) -> float:
        if len(equity_curve) == 0: return 0.0
        peak = equity_curve.cummax().iloc[-1]
        last = equity_curve.iloc[-1]
        return float(max((peak - last) / abs(peak), 0.0)) if peak != 0 else 0.0

    def _dd_to_scale(self, dd: float) -> float:
        for upper_bound, scale in self.thresholds:
            if dd <= upper_bound:
                return float(scale)
        return float(self.thresholds[-1][1])
