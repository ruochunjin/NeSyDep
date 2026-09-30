"""Vertical partitioning: project the relation onto correlated column sets.

Merges the sub-table generation logic of both prototypes
(``code-for-CFD/.../sampling/generate_sub_tables.py`` and the BSFD subsets).
"""
from __future__ import annotations

import pandas as pd

from nesydep.core.config import VerticalPartitionConfig
from nesydep.core.registry import PARTITIONERS
from nesydep.core.stages import CorrelationGraph, SubTable


@PARTITIONERS.decorator("vertical")
class VerticalPartitioner:
    """One sub-table per correlated set: columns = antecedents ∪ {target}."""

    def partition(
        self,
        frame: pd.DataFrame,
        graph: CorrelationGraph,
        config: VerticalPartitionConfig | None = None,
    ) -> list[SubTable]:
        cfg = config or VerticalPartitionConfig()
        subtables: list[SubTable] = []
        for i, cols in enumerate(graph.subtable_columns()):
            cols = tuple(c for c in cols if c in frame.columns)
            if cfg.max_columns_per_table is not None:
                cols = cols[: cfg.max_columns_per_table]
            if not cols:
                continue
            subtables.append(SubTable(frame=frame.loc[:, list(cols)], name=f"part{i}"))
        if not subtables:  # no correlated sets -> fall back to the full table
            subtables.append(SubTable(frame=frame, name="full"))
        return subtables
