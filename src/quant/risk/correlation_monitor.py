"""Correlation monitor — detects elevated inter-strategy correlation.

When avg pairwise correlation exceeds threshold, returns a reduced scale multiplier
to account for the collapsed diversification benefit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CorrelationMonitor:
    """Rolling average pairwise correlation across strategies."""

    def __init__(
        self,
        window_days: int = 30,
        alert_threshold: float = 0.7,
        high_corr_scale: float = 0.5,
        normal_scale: float = 1.0,
    ) -> None:
        if not 0 < alert_threshold <= 1:
            raise ValueError("alert_threshold must be in (0, 1]")
        self.window          = window_days
        self.threshold       = alert_threshold
        self.high_corr_scale = high_corr_scale
        self.normal_scale    = normal_scale

    def compute_avg_correlation(self, strategy_returns: pd.DataFrame) -> float:
        if strategy_returns.shape[1] < 2: return 0.0
        corr = strategy_returns.iloc[-self.window:].corr()
        return self._avg_pairwise(corr)

    def compute_scaling(self, strategy_returns: pd.DataFrame) -> float:
        avg_corr = self.compute_avg_correlation(strategy_returns)
        return float(self.high_corr_scale if avg_corr > self.threshold else self.normal_scale)

    def compute_scaling_series(self, strategy_returns: pd.DataFrame) -> pd.Series:
        index = strategy_returns.index
        scales = []
        for i in range(len(index)):
            if i < self.window - 1:
                scales.append(np.nan); continue
            window_df = strategy_returns.iloc[i - self.window + 1 : i + 1]
            avg = self._avg_pairwise(window_df.corr())
            scales.append(self.high_corr_scale if avg > self.threshold else self.normal_scale)
        return pd.Series(scales, index=index)

    def _avg_pairwise(self, corr_matrix: pd.DataFrame) -> float:
        n = corr_matrix.shape[0]
        if n < 2: return 0.0
        upper  = corr_matrix.values[np.triu_indices(n, k=1)]
        finite = upper[np.isfinite(upper)]
        return float(np.mean(finite)) if len(finite) > 0 else 0.0
