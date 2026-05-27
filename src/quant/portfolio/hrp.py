"""Hierarchical Risk Parity allocator — Lopez de Prado (2016), JPM.

Steps:
  1. Correlation matrix from recent returns
  2. Distance matrix = sqrt(0.5 * (1 - corr))
  3. Ward linkage hierarchical clustering
  4. Quasi-diagonalisation
  5. Recursive bisection with inverse-variance weights
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


class HRPAllocator:
    """Portfolio weights via Hierarchical Risk Parity."""

    def __init__(self, lookback_days: int = 252, rebalance_freq: str = "monthly") -> None:
        if lookback_days < 10:
            raise ValueError("lookback_days must be >= 10")
        self.lookback_days  = lookback_days
        self.rebalance_freq = rebalance_freq

    def compute_weights(self, strategy_returns: pd.DataFrame) -> dict[str, float]:
        """HRP weights without constraints. Returns dict summing to 1.0."""
        returns = self._prepare(strategy_returns)
        corr, cov = self._compute_matrices(returns)
        dist      = _distance_from_corr(corr)
        link      = _cluster(dist)
        sorted_idx = _quasi_diagonalize(link, len(returns.columns))
        sorted_names = [returns.columns[i] for i in sorted_idx]
        weights = self._recursive_bisection(cov, sorted_names)
        return {str(k): round(float(v), 6) for k, v in weights.items()}

    def compute_weights_with_constraints(
        self,
        returns: pd.DataFrame,
        min_weight: float = 0.05,
        max_weight: float = 0.40,
    ) -> dict[str, float]:
        """HRP with per-asset weight floor and cap (iterative water-filling)."""
        n = returns.shape[1]
        if min_weight * n > 1.0:
            raise ValueError(f"min_weight={min_weight} × {n} = {min_weight*n:.2f} > 1.0 — infeasible.")
        if max_weight * n < 1.0:
            raise ValueError(f"max_weight={max_weight} × {n} = {max_weight*n:.2f} < 1.0 — infeasible.")
        raw = self.compute_weights(returns)
        w   = _clip_and_renorm(pd.Series(raw), min_weight, max_weight)
        return {str(k): round(float(v), 6) for k, v in w.items()}

    def _prepare(self, strategy_returns: pd.DataFrame) -> pd.DataFrame:
        df = strategy_returns.iloc[-self.lookback_days:].copy()
        df = df.ffill().fillna(0.0)
        if df.shape[0] < 10:
            raise ValueError(f"Need >= 10 rows after lookback trim, got {df.shape[0]}.")
        if df.shape[1] < 2:
            raise ValueError("Need >= 2 strategy columns for HRP.")
        return df

    @staticmethod
    def _compute_matrices(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        cov  = returns.cov()
        std  = np.sqrt(np.maximum(np.diag(cov.values), 0.0))
        outer = np.outer(std, std)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr_vals = np.where(outer > 0, cov.values / outer, 0.0)
        np.fill_diagonal(corr_vals, 1.0)
        return pd.DataFrame(corr_vals, index=cov.index, columns=cov.columns), cov

    def _recursive_bisection(self, cov: pd.DataFrame, sorted_items: list[str]) -> pd.Series:
        weights  = pd.Series(1.0, index=sorted_items)
        clusters = [sorted_items]
        while clusters:
            next_clusters: list[list[str]] = []
            for cluster in clusters:
                if len(cluster) <= 1: continue
                mid   = len(cluster) // 2
                left, right = cluster[:mid], cluster[mid:]
                var_l = _get_cluster_var(cov, left)
                var_r = _get_cluster_var(cov, right)
                total = var_l + var_r
                alpha_l = (1.0 - var_l / total) if total > 0 else 0.5
                weights[left]  *= alpha_l
                weights[right] *= 1.0 - alpha_l
                next_clusters.extend([left, right])
            clusters = next_clusters
        return weights


def _distance_from_corr(corr: pd.DataFrame) -> np.ndarray:
    return np.sqrt(np.clip(0.5 * (1.0 - corr.values.clip(-1.0, 1.0)), 0.0, None))


def _cluster(dist: np.ndarray) -> np.ndarray:
    return linkage(squareform(dist, checks=False), method="ward")


def _quasi_diagonalize(link: np.ndarray, n: int) -> list[int]:
    link    = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    while sort_ix.max() >= n:
        sort_ix.index = range(0, len(sort_ix) * 2, 2)
        cluster_mask  = sort_ix >= n
        cluster_ids   = sort_ix[cluster_mask].values - n
        sort_ix[sort_ix.index[cluster_mask]] = link[cluster_ids, 0]
        right_children = pd.Series(link[cluster_ids, 1], index=sort_ix.index[cluster_mask] + 1)
        sort_ix = pd.concat([sort_ix, right_children]).sort_index().reset_index(drop=True)
    return sort_ix.tolist()


def _get_cluster_var(cov: pd.DataFrame, items: list[str]) -> float:
    sub_cov = cov.loc[items, items].values
    diag    = np.where(np.diag(sub_cov) > 0, np.diag(sub_cov), 1e-12)
    ivp     = 1.0 / diag; ivp /= ivp.sum()
    return float(ivp @ sub_cov @ ivp)


def _clip_and_renorm(weights: pd.Series, min_w: float, max_w: float, max_iter: int = 100) -> pd.Series:
    w = weights / weights.sum()
    for _ in range(max_iter):
        prev_w = w.copy()
        at_max, at_min = w >= max_w, w <= min_w
        free   = ~at_max & ~at_min
        w_new  = w.copy()
        w_new[at_max] = max_w; w_new[at_min] = min_w
        budget = 1.0 - w_new[at_max | at_min].sum()
        if free.any():
            free_sum = prev_w[free].sum()
            w_new[free] = budget * (prev_w[free] / free_sum) if free_sum > 1e-14 else budget / free.sum()
        else:
            return w_new / w_new.sum()
        if (w_new - prev_w).abs().max() < 1e-10:
            return w_new
        w = w_new
    return w
