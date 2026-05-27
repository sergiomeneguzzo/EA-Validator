"""Monte Carlo permutation test on trade P&L sequences."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class MCResult:
    n_simulations:      int
    p_profitable:       float
    dd_p5:              float
    dd_p50:             float
    dd_p95:             float
    final_equity_p5:    float
    final_equity_p50:   float
    final_equity_p95:   float
    expected_shortfall: float
    observed_max_dd:    float
    observed_final:     float
    all_max_dds:        np.ndarray
    all_finals:         np.ndarray


def mc_permutation_test(
    r_multiples: "list[float] | np.ndarray | pd.Series",
    n_simulations: int = 10_000,
    rng_seed: int | None = 42,
) -> MCResult:
    """Shuffle trade outcomes N times to separate edge from sequencing effect."""
    r = np.asarray(r_multiples, dtype=float)
    r = r[np.isfinite(r)]

    if len(r) < 2:
        raise ValueError("Need at least 2 finite R-multiples")

    rng = np.random.default_rng(rng_seed)

    obs_equity = np.cumsum(r)
    obs_final  = float(obs_equity[-1])
    obs_max_dd = _max_drawdown(obs_equity)

    finals:  np.ndarray = np.empty(n_simulations)
    max_dds: np.ndarray = np.empty(n_simulations)

    for i in range(n_simulations):
        perm       = rng.permutation(r)
        eq         = np.cumsum(perm)
        finals[i]  = eq[-1]
        max_dds[i] = _max_drawdown(eq)

    p_profitable = float(np.mean(finals > 0))
    cutoff = np.percentile(finals, 5)
    es     = float(finals[finals <= cutoff].mean()) if (finals <= cutoff).any() else float(cutoff)

    return MCResult(
        n_simulations      = n_simulations,
        p_profitable       = p_profitable,
        dd_p5              = float(np.percentile(max_dds, 5)),
        dd_p50             = float(np.percentile(max_dds, 50)),
        dd_p95             = float(np.percentile(max_dds, 95)),
        final_equity_p5    = float(np.percentile(finals, 5)),
        final_equity_p50   = float(np.percentile(finals, 50)),
        final_equity_p95   = float(np.percentile(finals, 95)),
        expected_shortfall = es,
        observed_max_dd    = obs_max_dd,
        observed_final     = obs_final,
        all_max_dds        = max_dds,
        all_finals         = finals,
    )


def plot_mc_distributions(result: MCResult, title: str = "") -> None:
    """Plot simulated max-DD and final-equity distributions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(result.all_max_dds, bins=60, color="#E53935", alpha=0.7, edgecolor="none")
    ax1.axvline(result.observed_max_dd, color="black", linewidth=1.5, linestyle="--",
                label=f"Observed = {result.observed_max_dd:.1f}")
    ax1.axvline(result.dd_p5,  color="#FF7043", linewidth=1, linestyle=":", label=f"p5  = {result.dd_p5:.1f}")
    ax1.axvline(result.dd_p95, color="#FF7043", linewidth=1, linestyle=":", label=f"p95 = {result.dd_p95:.1f}")
    ax1.set_xlabel("Max Drawdown"); ax1.set_ylabel("Frequency")
    ax1.set_title("Simulated Max Drawdown Distribution"); ax1.legend(fontsize=8)

    ax2.hist(result.all_finals, bins=60, color="#1E88E5", alpha=0.7, edgecolor="none")
    ax2.axvline(result.observed_final, color="black", linewidth=1.5, linestyle="--",
                label=f"Observed = {result.observed_final:.1f}")
    ax2.axvline(0, color="red", linewidth=1)
    ax2.set_xlabel("Final Equity"); ax2.set_ylabel("Frequency")
    ax2.set_title(f"Final Equity (P(profit)={result.p_profitable:.1%})"); ax2.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout(); plt.show()


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    hwm = np.maximum.accumulate(equity)
    return float(np.max(hwm - equity))
