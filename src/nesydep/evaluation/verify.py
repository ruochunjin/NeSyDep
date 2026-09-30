"""Verify CFDs/FDs against a full dataset: support & confidence.

Semantics (following the CFD literature and the prototypes' ``verify*.py``):

- *matching* a pattern position: equality for constant values; any value for
  the wildcard ``"_"``.
- ``support``   = rows matching the constant LHS positions **and** the RHS
  pattern.
- ``confidence`` = support / rows matching the constant LHS positions.

For variable CFDs (wildcard RHS) the embedded FD is additionally checked:
within rows matching the constant LHS positions, tuples agreeing on all LHS
attributes must agree on the RHS.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nesydep.core.dependency import CFD, FD, WILDCARD, Dependency


@dataclass(frozen=True)
class Verification:
    dependency: Dependency
    support: int
    confidence: float
    holds: bool  # confidence >= requested threshold (see verify())


def _lhs_mask(frame: pd.DataFrame, cfd: CFD) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for attr, pat in zip(cfd.lhs, cfd.lhs_pattern, strict=True):
        if pat != WILDCARD:
            mask &= frame[attr].astype("string") == pat
    return mask


def verify_cfd(frame: pd.DataFrame, cfd: CFD) -> tuple[int, float]:
    """Return ``(support, confidence)`` of ``cfd`` on ``frame``."""
    lhs_mask = _lhs_mask(frame, cfd)
    n_lhs = int(lhs_mask.sum())
    if n_lhs == 0:
        return 0, 0.0

    if cfd.rhs_pattern == WILDCARD:
        # Variable CFD: every row matches the RHS pattern; confidence reflects
        # how well the embedded FD holds among matching rows.
        sub = frame.loc[lhs_mask, list(cfd.lhs) + [cfd.rhs]]
        consistent = sub.groupby(list(cfd.lhs), sort=False)[cfd.rhs].nunique()
        if (consistent <= 1).all():
            return n_lhs, 1.0
        return n_lhs, 1.0 - _violation_fraction(sub, list(cfd.lhs), cfd.rhs)

    rhs_mask = frame[cfd.rhs].astype("string") == cfd.rhs_pattern
    support = int((lhs_mask & rhs_mask).sum())
    return support, support / n_lhs


def _violation_fraction(sub: pd.DataFrame, lhs: list[str], rhs: str) -> float:
    """Fraction of same-LHS groups that contain >1 distinct RHS value."""
    if sub.empty:
        return 0.0
    groups = sub.groupby(lhs, sort=False)[rhs].nunique()
    return float((groups > 1).sum() / len(groups)) if len(groups) else 0.0


def verify_fd(frame: pd.DataFrame, fd: FD) -> tuple[int, float]:
    """Return ``(support, confidence)`` of an FD (g3-style confidence).

    support    = number of rows in LHS groups that agree on the RHS.
    confidence = support / n_rows.
    """
    n = len(frame)
    if n == 0:
        return 0, 0.0
    lhs = list(fd.lhs)
    if not lhs:
        return n, 1.0 if frame[fd.rhs].nunique() <= 1 else 0.0
    sizes = frame.groupby(lhs, sort=False)[fd.rhs].nunique()
    consistent_groups = sizes[sizes <= 1]
    support = int(frame.groupby(lhs, sort=False).size()[consistent_groups.index].sum())
    return support, support / n


def verify(
    frame: pd.DataFrame,
    dependencies: list[Dependency],
    min_support: int = 1,
    min_confidence: float = 1.0,
) -> list[Verification]:
    """Verify each dependency; ``holds`` reflects the given thresholds."""
    out: list[Verification] = []
    for dep in dependencies:
        if isinstance(dep, CFD):
            sup, conf = verify_cfd(frame, dep)
        elif isinstance(dep, FD):
            sup, conf = verify_fd(frame, dep)
        else:
            raise TypeError(f"cannot verify dependency kind {dep.kind!r}")
        out.append(
            Verification(dep, sup, conf, holds=(sup >= min_support and conf >= min_confidence))
        )
    return out
