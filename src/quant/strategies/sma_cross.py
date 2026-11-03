from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def sma_cross(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    fast = df["close"].rolling(int(params.get("fast", 20))).mean()
    slow = df["close"].rolling(int(params.get("slow", 50))).mean()
    signal = pd.Series(np.where(fast > slow, 1, -1), index=df.index)
    return signal.where(slow.notna(), 0).shift(1).fillna(0).astype(int)
