"""Lightweight statistical correlation extractors (no GPU / no torch).

Refactored from the prototype ablation baselines
(``code-for-CFD/.../ablation/correlation/``). These give SCFDM a usable
correlation stage on machines without torch, trading some recall for a
fraction of the runtime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nesydep.core.config import LightweightCorrelationConfig
from nesydep.core.registry import CORRELATORS
from nesydep.core.stages import CorrelatedSet, CorrelationGraph


@CORRELATORS.decorator("lightweight")
class LightweightCorrelation:
    """Per-target feature ranking with Pearson / PCA / Lasso.

    For each target column Y, score every other column X and keep the
    ``max_antecedents`` highest-scoring ones above ``threshold``.
    """

    def extract(
        self, frame: pd.DataFrame, config: LightweightCorrelationConfig | None = None
    ) -> CorrelationGraph:
        cfg = config or LightweightCorrelationConfig()
        codes = self._encode(frame)
        scores = self._score(codes, cfg.method)

        sets: list[CorrelatedSet] = []
        cols = list(frame.columns)
        for j, target in enumerate(cols):
            order = np.argsort(-scores[j])
            antecedents = [cols[i] for i in order if i != j and scores[j, i] >= cfg.threshold][
                : cfg.max_antecedents
            ]
            if antecedents:
                sets.append(CorrelatedSet(antecedents=tuple(sorted(antecedents)), target=target))
        return CorrelationGraph(sets)

    @staticmethod
    def _encode(frame: pd.DataFrame) -> pd.DataFrame:
        """Ordinal-encode every column (correlation needs numeric input)."""
        return frame.apply(lambda s: pd.Categorical(s).codes)

    @staticmethod
    def _score(codes: pd.DataFrame, method: str) -> np.ndarray:
        """Return an (n_cols x n_cols) importance matrix; row = target."""
        if method == "pearson":
            corr = codes.corr(method="pearson").to_numpy()
            return np.nan_to_num(np.abs(corr))
        if method == "pca":
            from sklearn.decomposition import PCA

            n = min(codes.shape[1], codes.shape[0])
            loadings = np.abs(PCA(n_components=n).fit(codes).components_)
            # importance of feature i for "explaining" feature j is symmetric
            # here; use loading magnitude product as a soft association score.
            return loadings.T @ loadings
        if method == "lasso":
            from sklearn.linear_model import Lasso

            x = codes.to_numpy(dtype=float)
            n_cols = x.shape[1]
            out = np.zeros((n_cols, n_cols))
            for j in range(n_cols):
                others = [i for i in range(n_cols) if i != j]
                if not others:
                    continue
                model = Lasso(alpha=1e-4)  # paper default (prototype configuration.py)
                model.fit(x[:, others], x[:, j])
                for i, coef in zip(others, model.coef_, strict=True):
                    out[j, i] = abs(coef)
            return out
        raise ValueError(f"unknown lightweight correlation method {method!r}")
