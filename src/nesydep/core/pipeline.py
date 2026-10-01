"""Pipeline orchestration.

A :class:`Pipeline` wires the five stages together. Stages are addressed by
registry name, so users can swap a single stage (``correlation="pearson"``)
or pass pre-built instances.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nesydep.core.dataset import DatasetLike, as_categorical_view, as_dataset
from nesydep.core.registry import CORRELATORS, MINERS, PARTITIONERS, SAMPLERS
from nesydep.core.results import MiningResult
from nesydep.core.stages import (
    FullCorrelation,
    NoOpSampler,
    SingleTablePartitioner,
)


class _stage:
    """Per-stage progress: rich spinner on a TTY, plain lines otherwise."""

    def __init__(self, enabled: bool, message: str) -> None:
        self.enabled = enabled
        self.message = message
        self._status: Any = None

    def __enter__(self) -> "_stage":
        if not self.enabled:
            return self
        if sys.stderr.isatty():
            try:
                from rich.console import Console

                self._status = Console(stderr=True).status(self.message)
                self._status.__enter__()
                return self
            except ImportError:
                pass
        print(f"{self.message} ...", file=sys.stderr)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._status is not None:
            self._status.__exit__(*exc)


def _resolve(registry: Any, value: Any, identity: Any) -> Any:
    """Accept a registered name, an instance, or None (-> identity stage)."""
    if value is None:
        return identity
    if isinstance(value, str):
        return registry.get(value)()
    return value


@dataclass
class Pipeline:
    """Sample -> correlate -> partition -> mine.

    Attributes:
        sampler / correlation / partitioner / miner: registry name or instance.
        cache_dir: when set, intermediate artifacts (sample CSV, correlated
            sets, sub-table CSVs) are written there for reproducibility.
        configs: per-stage config objects keyed by stage name.
        progress: show per-stage progress (rich when installed and attached
            to a TTY, plain lines otherwise). False silences all output.
    """

    sampler: Any = None
    correlation: Any = None
    partitioner: Any = None
    miner: Any = None
    algorithm: str = "custom"
    cache_dir: str | Path | None = None
    configs: dict[str, Any] = field(default_factory=dict)
    global_config: Any = None
    progress: bool = True

    def run(self, data: DatasetLike | Any) -> MiningResult:
        dataset = as_dataset(data)
        frame = as_categorical_view(dataset)
        stats: dict[str, Any] = {"n_rows": len(frame), "n_columns": len(frame.columns)}

        sampler = _resolve(SAMPLERS, self.sampler, NoOpSampler())
        correlator = _resolve(CORRELATORS, self.correlation, FullCorrelation())
        partitioner = _resolve(PARTITIONERS, self.partitioner, SingleTablePartitioner())
        miner = _resolve(MINERS, self.miner, None)
        if miner is None:
            raise ValueError("Pipeline requires a miner (registry name or instance).")

        t0 = time.perf_counter()
        with _stage(self.progress, f"[1/4] sampling ({len(frame)} rows)"):
            sample = sampler.sample(frame, self.configs.get("sampler"))
        stats["sample_seconds"] = time.perf_counter() - t0
        stats["sample_rows"] = len(sample)

        t0 = time.perf_counter()
        with _stage(self.progress, "[2/4] correlation extraction"):
            graph = correlator.extract(sample, self.configs.get("correlation"))
        stats["correlation_seconds"] = time.perf_counter() - t0
        stats["correlated_sets"] = len(graph.sets)

        t0 = time.perf_counter()
        with _stage(self.progress, "[3/4] partitioning"):
            subtables = partitioner.partition(sample, graph, self.configs.get("partitioner"))
        stats["partition_seconds"] = time.perf_counter() - t0
        stats["n_subtables"] = len(subtables)

        self._write_cache(sample, graph, subtables)

        t0 = time.perf_counter()
        with _stage(self.progress, f"[4/4] mining ({len(subtables)} sub-tables)"):
            deps = miner.mine(subtables, self.configs.get("miner"))
        stats["mine_seconds"] = time.perf_counter() - t0

        return MiningResult(
            algorithm=self.algorithm,
            dependencies=list(deps),
            stats=stats,
            config_snapshot={
                k: (v.model_dump() if hasattr(v, "model_dump") else v)
                for k, v in self.configs.items()
            },
        )

    def _write_cache(self, sample: Any, graph: Any, subtables: Any) -> None:
        if self.cache_dir is None:
            return
        out = Path(self.cache_dir)
        out.mkdir(parents=True, exist_ok=True)
        sample.to_csv(out / "sample.csv", index=False)
        with open(out / "correlated_sets.txt", "w", encoding="utf-8") as f:
            for s in graph.sets:
                f.write(f"{{{', '.join(s.antecedents)}}} -> {s.target}\n")
        sub_dir = out / "subtables"
        sub_dir.mkdir(exist_ok=True)
        for i, st in enumerate(subtables):
            st.frame.to_csv(sub_dir / f"part{i}.csv", index=False)
