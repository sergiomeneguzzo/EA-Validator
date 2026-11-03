"""5-test validation scorecard — Lopez de Prado AFML.

Tests:
  1. purged_cv    : p25 OOS Sharpe > 0.5
  2. deflated_sr  : p_value < 0.05  (after correcting for n_trials)
  3. monte_carlo  : P(profit) > 0.7 over bootstrapped trade sequences  AND  p95 DD < 2× backtest DD
  4. robustness   : max_sensitivity < 0.50
  5. walk_forward : OOS/IS PF ratio > 0.70

Verdict:
  green  — ≥4 pass  → full capital allowed
  yellow — 3 pass   → paper trade only
  red    — ≤2 pass  → disabled
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

StrategyFn = Callable[[pd.DataFrame, dict[str, Any]], pd.Series]


@dataclass
class ScorecardConfig:
    purged_cv_p25_sr:    float = 0.5
    deflated_sr_p_value: float = 0.05
    mc_p_profitable:     float = 0.7
    mc_p5_dd_multiplier: float = 2.0
    robustness_max_sens: float = 0.50
    walk_forward_ratio:  float = 0.70
    n_trials_default:    int   = 10
    mc_simulations:      int   = 5_000
    cv_splits:           int   = 5
    deflated_sr_std:     float = 0.30
    cv_sharpe_method:    str   = "engine"


class StrategyScorecard:
    """Runs all 5 validation tests and returns a structured verdict."""

    def __init__(self, config: ScorecardConfig | None = None) -> None:
        self.cfg = config or ScorecardConfig()

    def evaluate(
        self,
        strategy_name: str,
        strategy_fn:   StrategyFn,
        df_ohlcv:      pd.DataFrame,
        base_params:   dict[str, Any],
        param_ranges:  dict[str, tuple] | None = None,
        n_trials:      int | None = None,
    ) -> dict:
        """Run all 5 tests. Returns nested dict with per-test results + overall_status."""
        from quant.backtest.engine import run_backtest
        from quant.validation.purged_cv       import cross_val_score_purged
        from quant.validation.deflated_sharpe import annualise_sharpe, deflated_sharpe, sharpe_from_returns
        from quant.validation.monte_carlo     import mc_bootstrap_test
        from quant.validation.robustness      import robustness_test
        from quant.validation.walk_forward    import walk_forward_test

        cfg = self.cfg
        _nt = n_trials or cfg.n_trials_default
        results: dict[str, Any] = {"strategy": strategy_name}

        try:
            _raw = strategy_fn(df_ohlcv, base_params)
            if isinstance(_raw, tuple):
                base_sig = _raw[0]
                _base_sl = _raw[1] if len(_raw) > 1 else None
                _base_tp = _raw[2] if len(_raw) > 2 else None
            else:
                base_sig, _base_sl, _base_tp = _raw, None, None
            base_result = run_backtest(df_ohlcv, base_sig, sl_prices=_base_sl, tp_prices=_base_tp)
            base_trades = base_result["trades"]
            base_m      = base_result["metrics"]
            base_pf     = base_m["profit_factor"]
            base_max_dd = base_m["max_dd"]
        except Exception as exc:
            log.error("Baseline backtest failed for %s: %s", strategy_name, exc)
            return self._failed_result(strategy_name, str(exc))

        # 1. Purged CV
        try:
            oos_sharpes = cross_val_score_purged(
                lambda df: strategy_fn(df, base_params),
                df_ohlcv, n_splits=cfg.cv_splits, sharpe_method=cfg.cv_sharpe_method,
            )
            valid_sr = [s for s in oos_sharpes if not np.isnan(s)]
            mean_sr  = float(np.mean(valid_sr))       if valid_sr else float("nan")
            std_sr   = float(np.std(valid_sr))        if valid_sr else float("nan")
            p25_sr   = float(np.percentile(valid_sr, 25)) if valid_sr else float("nan")
            cv_pass  = not np.isnan(p25_sr) and p25_sr > cfg.purged_cv_p25_sr
        except Exception as exc:
            log.warning("Purged CV failed for %s: %s", strategy_name, exc)
            mean_sr = std_sr = p25_sr = float("nan"); cv_pass = False; oos_sharpes = []

        results["purged_cv"] = {
            "oos_sharpes": oos_sharpes, "mean_sr": round(mean_sr, 3),
            "std_sr": round(std_sr, 3), "p25_sr": round(p25_sr, 3), "pass": bool(cv_pass),
        }

        # 2. Deflated Sharpe
        try:
            daily_pnl = _trades_to_daily_pnl(base_trades, df_ohlcv.index)
            sr_daily, skew, exc_kurt, n_obs = sharpe_from_returns(daily_pnl)
            dsr, p_val = deflated_sharpe(sr_daily, n_trials=_nt, skewness=skew, excess_kurtosis=exc_kurt,
                                         n_obs=n_obs, sr_std=cfg.deflated_sr_std / math.sqrt(252))
            sr_hat = annualise_sharpe(sr_daily)
            dsr_pass = not np.isnan(p_val) and p_val < cfg.deflated_sr_p_value
        except Exception as exc:
            log.warning("Deflated SR failed for %s: %s", strategy_name, exc)
            sr_hat = dsr = p_val = float("nan"); dsr_pass = False

        results["deflated_sr"] = {
            "sr_hat": round(sr_hat, 3), "dsr": round(dsr, 3),
            "p_value": round(p_val, 4), "n_trials": _nt, "pass": bool(dsr_pass),
        }

        # 3. Monte Carlo
        try:
            r_mults = base_trades["pnl"].dropna().values
            mc      = mc_bootstrap_test(r_mults, n_simulations=cfg.mc_simulations)
            mc_dd_pass = mc.dd_p95 <= cfg.mc_p5_dd_multiplier * max(mc.observed_max_dd, 1e-9)
            mc_pass    = mc.p_profitable >= cfg.mc_p_profitable and mc_dd_pass
        except Exception as exc:
            log.warning("Monte Carlo failed for %s: %s", strategy_name, exc)
            mc = None; mc_pass = False

        if mc is not None:
            results["monte_carlo"] = {
                "p_profitable": round(mc.p_profitable, 3), "dd_p5": round(mc.dd_p5, 2),
                "dd_p50": round(mc.dd_p50, 2), "dd_p95": round(mc.dd_p95, 2),
                "final_p50": round(mc.final_equity_p50, 2),
                "expected_shortfall": round(mc.expected_shortfall, 2), "pass": bool(mc_pass),
            }
        else:
            results["monte_carlo"] = {"pass": False, "error": "failed"}

        # 4. Robustness
        try:
            rob_df     = robustness_test(strategy_fn, df_ohlcv, base_params, param_ranges=param_ranges)
            max_sens   = float(rob_df["max_sensitivity"].max()) if len(rob_df) else 0.0
            fragile    = rob_df.loc[rob_df["fragile"], "param"].tolist()
            rob_pass   = not np.isnan(max_sens) and max_sens < cfg.robustness_max_sens
        except Exception as exc:
            log.warning("Robustness failed for %s: %s", strategy_name, exc)
            max_sens = float("nan"); fragile = []; rob_pass = False

        results["robustness"] = {
            "max_sensitivity": round(max_sens, 3), "fragile_params": fragile, "pass": bool(rob_pass),
        }

        # 5. Walk-forward
        try:
            wf = walk_forward_test(strategy_fn, df_ohlcv, base_params)
        except Exception as exc:
            log.warning("Walk-forward failed for %s: %s", strategy_name, exc)
            wf = {"is_pf": float("nan"), "oos_pf": float("nan"), "is_oos_ratio": float("nan"), "pass": False}

        results["walk_forward"] = {
            "is_pf": round(wf["is_pf"], 3), "val_pf": round(wf.get("val_pf", float("nan")), 3),
            "oos_pf": round(wf["oos_pf"], 3),
            "is_oos_ratio": round(wf["is_oos_ratio"], 3) if not np.isnan(wf["is_oos_ratio"]) else float("nan"),
            "pass": bool(wf["pass"]),
        }

        n_pass = sum(results[t]["pass"] for t in ("purged_cv", "deflated_sr", "monte_carlo", "robustness", "walk_forward"))
        status = "green" if n_pass >= 4 else ("yellow" if n_pass == 3 else "red")

        results["overall_status"]  = status
        results["n_tests_passed"]  = n_pass
        results["capital_allowed"] = status == "green"
        results["base_pf"]         = round(base_pf, 3)
        results["base_sharpe"]     = round(base_m["sharpe"], 3)
        results["base_max_dd"]     = round(base_max_dd, 4)
        results["base_n_trades"]   = base_m["n_trades"]
        return results

    def _failed_result(self, name: str, error: str) -> dict:
        return {"strategy": name, "overall_status": "red", "n_tests_passed": 0,
                "capital_allowed": False, "error": error}


def _trades_to_daily_pnl(trades: pd.DataFrame, full_index: pd.DatetimeIndex) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    t = trades.copy()
    t["date"] = pd.to_datetime(t["exit_time"]).dt.normalize()
    daily = t.groupby("date")["pnl"].sum()
    trading_days = pd.DatetimeIndex(pd.to_datetime(full_index).normalize().unique())
    return daily.reindex(trading_days.union(daily.index), fill_value=0.0)
