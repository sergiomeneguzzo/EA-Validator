"""3-state Gaussian HMM regime detector (bull / range / crisis)."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logging.getLogger("hmmlearn").setLevel(logging.ERROR)


class HMMRegimeDetector:
    """Gaussian HMM on log-returns + realized volatility features."""

    def __init__(self, n_states: int = 3, features: list[str] | None = None) -> None:
        self.n_states = n_states
        self.features: list[str] = features if features is not None else ["returns", "log_vol"]
        self._model:        GaussianHMM | None = None
        self._state_labels: dict[int, str] | None = None

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        cols: dict[str, pd.Series] = {}
        if "returns"  in self.features: cols["returns"]  = log_ret
        if "log_vol"  in self.features:
            rvol = log_ret.rolling(20).std() * np.sqrt(252)
            cols["log_vol"] = np.log(rvol.clip(lower=1e-8))
        return pd.DataFrame(cols, index=df.index).dropna()

    def fit(self, df: pd.DataFrame) -> None:
        """Fit HMM. Recommend at least 500 bars for stable convergence."""
        feat_df = self._build_features(df)
        X = feat_df.values.astype(np.float64)
        self._model = GaussianHMM(n_components=self.n_states, covariance_type="full",
                                   n_iter=200, tol=1e-4, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X)
        self._state_labels = self.label_states()

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smoothed posterior probabilities. Returns columns p_state_N and p_bull/p_range/p_crisis."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        feat_df    = self._build_features(df)
        posteriors = self._model.predict_proba(feat_df.values.astype(np.float64))
        result = pd.DataFrame(
            {f"p_state_{i}": posteriors[:, i] for i in range(self.n_states)},
            index=feat_df.index,
        )
        if self._state_labels is not None:
            inv = {v: k for k, v in self._state_labels.items()}
            for label in ("bull", "range", "crisis"):
                state_idx = inv.get(label)
                if state_idx is not None:
                    result[f"p_{label}"] = result[f"p_state_{state_idx}"]
        return result

    def label_states(self) -> dict[int, str]:
        """Identify bull (highest return), crisis (highest vol), range (remainder)."""
        if self._model is None:
            raise RuntimeError("Call fit() before label_states().")
        means     = self._model.means_
        ret_col   = self.features.index("returns") if "returns" in self.features else 0
        bull_state = int(np.argmax(means[:, ret_col]))
        labels     = {bull_state: "bull"}

        if "log_vol" in self.features:
            vol_col = self.features.index("log_vol")
            vol_copy = means[:, vol_col].copy()
            vol_copy[bull_state] = -np.inf
            crisis_state = int(np.argmax(vol_copy))
        else:
            ret_copy = means[:, ret_col].copy().astype(float)
            ret_copy[bull_state] = np.inf
            crisis_state = int(np.argmin(ret_copy))

        labels[crisis_state] = "crisis"
        range_states = [i for i in range(self.n_states) if i not in labels]
        labels[range_states[0] if range_states else bull_state] = "range"
        self._state_labels = labels
        return labels

    @property
    def state_labels(self) -> dict[int, str] | None:
        return self._state_labels

    def predict_dominant(self, df: pd.DataFrame) -> pd.Series:
        """Most probable semantic label per bar: 'bull' / 'range' / 'crisis'."""
        proba = self.predict_proba(df)
        if self._state_labels is None:
            raise RuntimeError("label_states() must be called first.")
        raw_cols = [f"p_state_{i}" for i in range(self.n_states)]
        dominant  = proba[raw_cols].values.argmax(axis=1)
        return pd.Series([self._state_labels[i] for i in dominant], index=proba.index, name="regime")
