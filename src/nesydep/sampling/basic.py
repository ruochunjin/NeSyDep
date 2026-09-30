"""Row-level samplers."""
from __future__ import annotations

import pandas as pd

from nesydep.core.config import (
    RandomSampleConfig,
    RepresentativeSampleConfig,
    StratifiedSampleConfig,
)
from nesydep.core.registry import SAMPLERS


@SAMPLERS.decorator("random")
class RandomSampler:
    """Uniform random sampling without replacement."""

    def sample(self, frame: pd.DataFrame, config: RandomSampleConfig | None = None) -> pd.DataFrame:
        cfg = config or RandomSampleConfig()
        n = max(1, round(len(frame) * cfg.ratio))
        return frame.sample(n=min(n, len(frame)), random_state=cfg.seed).reset_index(drop=True)


@SAMPLERS.decorator("stratified")
class StratifiedSampler:
    """Stratified sampling (BSFD).

    Rows are sampled proportionally within the strata of a chosen column —
    by default the column with the most distinct values, matching the
    prototype's behaviour (``code-for-BSFD/sample/sampler.py``).
    """

    def sample(
        self, frame: pd.DataFrame, config: StratifiedSampleConfig | None = None
    ) -> pd.DataFrame:
        cfg = config or StratifiedSampleConfig()
        if len(frame) == 0:
            return frame
        stratify_by = cfg.stratify_by
        if stratify_by is None:
            stratify_by = max(frame.columns, key=lambda c: frame[c].nunique())
        parts = [
            g.sample(n=max(1, round(len(g) * cfg.ratio)), random_state=cfg.seed)
            for _, g in frame.groupby(stratify_by)
        ]
        return pd.concat(parts).reset_index(drop=True)
