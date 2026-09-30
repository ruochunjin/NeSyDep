"""RepSampler: MinHash-LSH diversity-first representative sampling (SCFDM).

Faithful port of the prototype
(``code-for-CFD/correlation-and-sampling/sampling/representative_tuple_sampling.py``):

1. Stream-scan rows; hash each row's ``col_value`` features with MinHash and
   bucket by the first few hash values.
2. Sort buckets ascending by size; round-robin pick one row per bucket
   (diversity-first) until the bound is reached.
3. Materialise the picked rows.

Requires the ``lsh`` extra (``datasketch``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nesydep.core.config import RepresentativeSampleConfig
from nesydep.core.registry import SAMPLERS
from nesydep.sampling.bounds import p_hat_estimate, theorem2_bound

BUCKET_PREFIX = 4  # number of MinHash values used as the bucket key (paper value)


@SAMPLERS.decorator("representative")
class RepSampler:
    """Diversity-first representative sampling via MinHash-LSH."""

    def sample(
        self, frame: pd.DataFrame, config: RepresentativeSampleConfig | None = None
    ) -> pd.DataFrame:
        cfg = config or RepresentativeSampleConfig()
        try:
            from datasketch import MinHash
        except ImportError as e:
            raise ImportError(
                "RepSampler requires the 'lsh' extra: pip install 'nesydep[lsh]'"
            ) from e

        n_bound = cfg.bound
        if n_bound == "auto":
            # Paper §7.1: estimate p̂ from the data, then apply Theorem 2.
            p_hat = p_hat_estimate(frame, delta_hat=2e-3)
            n_bound = theorem2_bound(p_hat, cfg.eps, cfg.eta, cfg.lam)
        n_bound = min(int(n_bound), len(frame))
        if n_bound >= len(frame):
            return frame.reset_index(drop=True)

        rng = np.random.default_rng(cfg.seed)
        columns = list(frame.columns)
        values = frame.astype("string").fillna("").to_numpy()

        # 1. Bucket row indices by MinHash signature prefix.
        buckets: dict[tuple, list[int]] = {}
        for idx in range(len(frame)):
            m = MinHash(num_perm=cfg.num_perm)
            row = values[idx]
            for col, val in zip(columns, row, strict=True):
                m.update(f"{col}_{val}".encode())
            key = tuple(m.hashvalues[:BUCKET_PREFIX])
            buckets.setdefault(key, []).append(idx)

        # 2. Diversity-first round-robin selection.
        bucket_list = [buckets[k] for k in sorted(buckets, key=lambda k: len(buckets[k]))]
        chosen: set[int] = set()
        while len(chosen) < n_bound and bucket_list:
            to_remove = []
            for i, bucket in enumerate(bucket_list):
                if len(chosen) >= n_bound:
                    break
                if bucket:
                    pos = int(rng.integers(len(bucket)))
                    chosen.add(bucket.pop(pos))
                else:
                    to_remove.append(i)
            for i in reversed(to_remove):
                bucket_list.pop(i)

        # 3. Materialise.
        return frame.iloc[sorted(chosen)].reset_index(drop=True)
