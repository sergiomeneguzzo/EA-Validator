"""Parameter robustness / sensitivity analysis.

Perturbs each numeric parameter ±perturbation_pct and measures PF sensitivity.
Sensitivity > 0.5 (PF halves with ±20% param change) → fragile strategy.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

StrategyFn = Callable[[pd.DataFrame, dict[str, Any]], pd.Series]


def robustness_test(
    strategy_fn: StrategyFn,
    df_ohlcv: pd.DataFrame,
    base_params: dict[str, Any],
    param_ranges: dict[str, tuple[Any, Any]] | None = None,
    perturbation_pct: float = 0.20,
) -> pd.DataFrame:
    """Sensitivity analysis: perturb each numeric parameter ±perturbation_pct.

    Returns DataFrame with columns: param, base_value, pf_base, pf_up, pf_down,
    sensitivity_up, sensitivity_down, max_sensitivity, fragile.
    """
    from quant.backtest.engine import run_backtest

    def _run(params: dict) -> float:
        try:
            _raw = strategy_fn(df_ohlcv, params)
            if isinstance(_raw, tuple):
                sig, sl, tp = _raw[0], (_raw[1] if len(_raw) > 1 else None), (_raw[2] if len(_raw) > 2 else None)
            else:
                sig, sl, tp = _raw, None, None
            return float(run_backtest(df_ohlcv, sig, sl_prices=sl, tp_prices=tp)["metrics"]["profit_factor"])
        except Exception:
            return float("nan")

    pf_base = _run(base_params)
    rows: list[dict] = []

    for param, base_val in base_params.items():
        if not isinstance(base_val, (int, float)) or isinstance(base_val, bool):
            continue

        if param_ranges and param in param_ranges:
            lo, hi = param_ranges[param]
        else:
            delta = abs(base_val) * perturbation_pct if base_val != 0 else perturbation_pct
            lo, hi = base_val - delta, base_val + delta

        if isinstance(base_val, int):
            lo = max(1, int(round(lo)))
            hi = max(lo + 1, int(round(hi)))

        pf_up   = _run({**base_params, param: hi})
        pf_down = _run({**base_params, param: lo})

        def _sens(pf_other: float) -> float:
            if np.isnan(pf_base) or pf_base == 0 or np.isnan(pf_other):
                return float("nan")
            return abs(pf_other - pf_base) / pf_base

        sens_up   = _sens(pf_up)
        sens_down = _sens(pf_down)
        max_sens  = float(np.nanmax([sens_up, sens_down]))

        rows.append({
            "param":            param,
            "base_value":       base_val,
            "up_value":         hi,
            "down_value":       lo,
            "pf_base":          round(pf_base,   3),
            "pf_up":            round(pf_up,     3) if not np.isnan(pf_up)   else float("nan"),
            "pf_down":          round(pf_down,   3) if not np.isnan(pf_down) else float("nan"),
            "sensitivity_up":   round(sens_up,   3) if not np.isnan(sens_up)   else float("nan"),
            "sensitivity_down": round(sens_down, 3) if not np.isnan(sens_down) else float("nan"),
            "max_sensitivity":  round(max_sens,  3),
            "fragile":          max_sens > 0.50,
        })

    if not rows:
        return pd.DataFrame(columns=["param", "base_value", "up_value", "down_value",
                                      "pf_base", "pf_up", "pf_down", "sensitivity_up",
                                      "sensitivity_down", "max_sensitivity", "fragile"])
    return pd.DataFrame(rows).sort_values("max_sensitivity", ascending=False)
