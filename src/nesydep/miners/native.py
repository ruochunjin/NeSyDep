"""Wrappers around the native C++ mining kernels (``nesydep._core``).

The extension is built from ``src/cpp`` via scikit-build-core. When it is not
available (no compiler on the machine), importing this module still works;
instantiating a native miner raises a clear error at mining time.

Sub-table-level parallelism lives here in Python (``ProcessPoolExecutor``),
replacing the prototypes' MPI layer. OpenMP inside the kernels is configured
through the config's ``openmp_threads``.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any

from nesydep.core.config import CFDMinerConfig, FDMinerConfig
from nesydep.core.dependency import CFD, FD, Dependency
from nesydep.core.registry import MINERS
from nesydep.core.stages import SubTable


def _core() -> Any:
    try:
        from nesydep import _core

        return _core
    except ImportError as e:
        raise ImportError(
            "the native mining kernels are not available. Reinstall with a C++ "
            "toolchain present (pip install nesydep), or use the pure-Python "
            "reference miner 'pyref-fd' for small inputs."
        ) from e


def _frame_payload(st: SubTable) -> tuple[list[str], list[list[str]]]:
    return [str(c) for c in st.frame.columns], st.frame.astype(str).values.tolist()


def _run_parallel(
    call: _NativeCall,
    subtables: list[SubTable],
    n_jobs: int,
) -> list:
    if n_jobs <= 1 or len(subtables) <= 1:
        return [d for st in subtables for d in call(*_frame_payload(st))]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        results = list(pool.map(call.for_payload, [_frame_payload(st) for st in subtables]))
    return [d for batch in results for d in batch]


class _NativeCall:
    """Picklable callable: apply a native kernel to one (columns, rows) payload."""

    def __init__(self, func_name: str, params: dict[str, Any]) -> None:
        self.func_name = func_name
        self.params = params

    def __call__(self, columns: list[str], rows: list[list[str]]) -> list:
        return getattr(_core(), self.func_name)(columns, rows, **self.params)

    def for_payload(self, payload: tuple[list[str], list[list[str]]]) -> list:
        return self(*payload)


def _n_jobs(config: Any) -> int:
    return int(getattr(config, "n_jobs", 1))


class SearchSpaceExplosionError(RuntimeError):
    """Raised instead of an opaque OOM when level-wise mining would explode."""


def _estimate_candidates(n_cols: int, max_lhs: int) -> int:
    """Worst-case level-wise candidate sets: sum of C(n, k) over levels.

    Level k holds k-attribute sets plus their candidate RHS assignments;
    with max_lhs the lattice is capped at level max_lhs + 1.
    """
    from math import comb

    top = n_cols if max_lhs <= 0 else min(n_cols, max_lhs + 1)
    return sum(comb(n_cols, k) for k in range(1, top + 1))


def _guard_search_space(subtables: list[SubTable], max_columns: int, max_lhs: int = 0,
                        budget: int = 10_000_000) -> None:
    """Refuse level-wise mining before it OOMs.

    The TANE/DFD kernels enumerate the attribute lattice level by level; with
    unbounded LHS the space is 2^n and exhausts memory beyond ~20 columns.
    A finite max_lhs caps the lattice at level max_lhs+1, which the estimate
    accounts for. Wide unbounded tables are exactly what the partitioned
    pipelines (bsfd/scfdm) exist for.
    """
    if max_columns <= 0:
        return
    for st in subtables:
        n = len(st.frame.columns)
        estimated = _estimate_candidates(n, max_lhs)
        if n > max_columns and estimated > budget:
            raise SearchSpaceExplosionError(
                f"sub-table {st.name!r} has {n} columns (est. {estimated:,} "
                f"lattice candidates, budget {budget:,}). Level-wise exact "
                "mining would exhaust memory. Use a partitioned pipeline "
                "(algo='bsfd' / 'scfdm'), set max_lhs to bound the lattice, or "
                "raise max_columns_guard if you know what you are doing."
            )


def _mine_safely(call: _NativeCall, subtables: list[SubTable], n_jobs: int) -> list:
    try:
        return _run_parallel(call, subtables, n_jobs)
    except MemoryError as e:
        raise SearchSpaceExplosionError(
            "the native miner ran out of memory. The search space is too large "
            "for this input: reduce columns (partitioned pipeline), raise "
            "support/confidence thresholds, or lower max_lhs."
        ) from e


class _NativeFDMinerBase:
    """FD kernels return ``[(lhs_columns, rhs_column), ...]``."""

    func_name = ""

    def mine(self, subtables: list[SubTable], config: FDMinerConfig | None = None) -> list[FD]:
        cfg = config or FDMinerConfig()
        _guard_search_space(subtables, cfg.max_columns_guard, cfg.max_lhs)
        call = _NativeCall(
            self.func_name,
            {
                "support": cfg.support,
                "confidence": cfg.confidence,
                "max_lhs": cfg.max_lhs,
            },
        )
        raw = _mine_safely(call, subtables, _n_jobs(cfg))
        return [FD(lhs=tuple(lhs), rhs=rhs) for lhs, rhs in raw]


@MINERS.decorator("pfminer")
class PFMinerNative(_NativeFDMinerBase):
    """BSFD's parallel FD kernel (native; per-sub-table TANE inside the framework)."""

    func_name = "pfminer_mine"


@MINERS.decorator("tane")
class TaneNative(_NativeFDMinerBase):
    """Classic TANE baseline (native)."""

    func_name = "tane_mine"


@MINERS.decorator("dfd")
class DFDNative(_NativeFDMinerBase):
    """Classic DFD baseline (native)."""

    func_name = "dfd_mine"


@MINERS.decorator("scfdm")
class SCFDMNative:
    """SCFDM CFD kernel (native). 10 search strategies; CTane = 'Integrated-BFS'."""

    def mine(
        self, subtables: list[SubTable], config: CFDMinerConfig | None = None
    ) -> list[Dependency]:
        cfg = config or CFDMinerConfig()
        _guard_search_space(subtables, cfg.max_columns_guard, cfg.max_lhs)
        call = _NativeCall(
            "cfd_mine",
            {
                "support": cfg.support,
                "confidence": cfg.confidence,
                "max_lhs": cfg.max_lhs,
                "strategy": cfg.strategy,
                "constant_only": cfg.constant_only,
            },
        )
        raw = _mine_safely(call, subtables, _n_jobs(cfg))
        return [
            CFD(
                lhs=tuple(lhs),
                rhs=rhs,
                lhs_pattern=tuple(pattern),
                rhs_pattern=rhs_pattern,
            )
            for lhs, rhs, pattern, rhs_pattern in raw
        ]


@MINERS.decorator("ctane")
class CTaneNative(SCFDMNative):
    """CTane baseline — SCFDM's 'Integrated-BFS' strategy."""

    def mine(
        self, subtables: list[SubTable], config: CFDMinerConfig | None = None
    ) -> list[Dependency]:
        cfg = (config or CFDMinerConfig()).model_copy(update={"strategy": "Integrated-BFS"})
        return super().mine(subtables, cfg)
