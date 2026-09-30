"""Pure-Python reference miners.

Correct-but-simple implementations used for:
- running the pipeline end-to-end where the C++ extension is unavailable
  (e.g. development machines without a toolchain);
- cross-checking the native kernels in tests on small inputs.

Not optimised — do not use on large data.
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from nesydep.core.config import FDMinerConfig
from nesydep.core.dependency import FD
from nesydep.core.registry import MINERS
from nesydep.core.stages import SubTable


def _holds(frame: pd.DataFrame, lhs: tuple[str, ...], rhs: str) -> bool:
    if not lhs:
        return frame[rhs].nunique() <= 1
    return bool((frame.groupby(list(lhs), sort=False)[rhs].nunique() <= 1).all())


def discover_fds_naive(
    frame: pd.DataFrame, max_lhs: int = 0, include_constant_rhs: bool = False
) -> list[FD]:
    """Minimal non-trivial FDs by level-wise LHS enumeration (exact)."""
    cols = list(frame.columns)
    results: list[FD] = []
    for rhs in cols:
        if frame[rhs].nunique() <= 1 and not include_constant_rhs:
            # constant RHS: every LHS works; the minimal FD is ()->rhs which
            # TANE-style miners report as trivial — skip to match.
            continue
        others = [c for c in cols if c != rhs]
        found: list[tuple[str, ...]] = []
        limit = max_lhs if max_lhs > 0 else len(others)
        for k in range(1, limit + 1):
            for combo in combinations(others, k):
                if any(set(f) <= set(combo) for f in found):
                    continue  # not minimal
                if _holds(frame, combo, rhs):
                    found.append(combo)
                    results.append(FD(lhs=combo, rhs=rhs))
    return results


@MINERS.decorator("pyref-fd")
class NaiveFDMiner:
    """Reference FD miner over each sub-table (union of per-table FDs)."""

    def mine(self, subtables: list[SubTable], config: FDMinerConfig | None = None) -> list[FD]:
        cfg = config or FDMinerConfig()
        out: set[FD] = set()
        for st in subtables:
            out.update(discover_fds_naive(st.frame, max_lhs=cfg.max_lhs))
        return sorted(out, key=str)
