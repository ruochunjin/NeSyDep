"""Precision / recall / F1 for discovered dependencies.

Set-level comparison over normalised keys, matching both prototype
evaluation modules (``effectiveness_measure.py`` for FDs,
``supplemental/evaluation/metrics.py`` for CFDs).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from nesydep.core.dependency import Dependency
from nesydep.io.legacy import normalize_key


@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    f1: float
    n_mined: int
    n_ground_truth: int
    n_correct: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "n_mined": self.n_mined,
            "n_ground_truth": self.n_ground_truth,
            "n_correct": self.n_correct,
        }


def evaluate(
    mined: Iterable[Dependency],
    ground_truth: Iterable[Dependency],
) -> Metrics:
    """Set-based P/R/F1 between mined and ground-truth dependencies.

    Comparison uses :func:`nesydep.io.legacy.normalize_key`, so LHS order and
    pattern ordering do not affect the result. Kinds never mix: an FD never
    matches a CFD.
    """
    mined_keys = {normalize_key(d) for d in mined}
    truth_keys = {normalize_key(d) for d in ground_truth}
    correct = mined_keys & truth_keys

    precision = len(correct) / len(mined_keys) if mined_keys else 0.0
    recall = len(correct) / len(truth_keys) if truth_keys else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        n_mined=len(mined_keys),
        n_ground_truth=len(truth_keys),
        n_correct=len(correct),
    )
