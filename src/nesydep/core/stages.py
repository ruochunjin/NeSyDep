"""Pipeline stage contracts.

Five protocols model the pipeline shared by every fast-dependency-discovery
method we know of::

    Sampler  ->  CorrelationExtractor  ->  Partitioner  ->  Miner  ->  Evaluator
    (rows)        (column groups)         (sub-tables)      (deps)     (scores)

Algorithms plug in by implementing the protocols they need and registering
under a name (see :mod:`nesydep.core.registry`). End-to-end baselines
(TANE, DFD, ...) implement only :class:`Miner`; the pipeline substitutes
identity stages for the rest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from nesydep.core.dependency import Dependency
from nesydep.core.results import MiningResult


@dataclass(frozen=True)
class CorrelatedSet:
    """One correlated attribute group: ``antecedents -> target``.

    Partitions are built by projecting the relation onto
    ``antecedents ∪ {target}``.
    """

    antecedents: tuple[str, ...]
    target: str

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.antecedents) | {self.target}))


@dataclass
class CorrelationGraph:
    """The output of correlation extraction: a list of correlated sets."""

    sets: list[CorrelatedSet]

    def subtable_columns(self) -> list[tuple[str, ...]]:
        """One column tuple per sub-table, duplicates merged."""
        seen: set[tuple[str, ...]] = set()
        out: list[tuple[str, ...]] = []
        for s in self.sets:
            if s.columns not in seen:
                seen.add(s.columns)
                out.append(s.columns)
        return out


@dataclass
class SubTable:
    """A vertical slice of the relation, mined independently."""

    frame: pd.DataFrame  # all-string view (see dataset.as_categorical_view)
    name: str = "subtable"


@runtime_checkable
class Sampler(Protocol):
    """Row-level reduction."""

    def sample(self, frame: pd.DataFrame, config: Any) -> pd.DataFrame: ...


@runtime_checkable
class CorrelationExtractor(Protocol):
    """Column-level reduction: find groups of correlated attributes."""

    def extract(self, frame: pd.DataFrame, config: Any) -> CorrelationGraph: ...


@runtime_checkable
class Partitioner(Protocol):
    """Project the (sampled) relation into sub-tables."""

    def partition(self, frame: pd.DataFrame, graph: CorrelationGraph, config: Any) -> list[SubTable]: ...


@runtime_checkable
class Miner(Protocol):
    """Discover dependencies on one or more sub-tables."""

    def mine(self, subtables: list[SubTable], config: Any) -> list[Dependency]: ...


@runtime_checkable
class Evaluator(Protocol):
    """Score a result against ground truth."""

    def evaluate(self, result: MiningResult, ground_truth: list[Dependency]) -> dict[str, float]: ...


# -- identity stages (used by end-to-end baseline algorithms) -----------------


class NoOpSampler:
    def sample(self, frame: pd.DataFrame, config: Any) -> pd.DataFrame:
        return frame


class FullCorrelation:
    """Treat every column as potentially correlated (no reduction)."""

    def extract(self, frame: pd.DataFrame, config: Any) -> CorrelationGraph:
        cols = [str(c) for c in frame.columns]
        return CorrelationGraph([CorrelatedSet(antecedents=tuple(cols), target=cols[0])])


class SingleTablePartitioner:
    """One sub-table = the whole frame (no vertical split)."""

    def partition(self, frame: pd.DataFrame, graph: CorrelationGraph, config: Any) -> list[SubTable]:
        return [SubTable(frame=frame, name="full")]
