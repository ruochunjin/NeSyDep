"""Wrappers around the native C++ mining kernels (``nesydep._core``).

The extension is built from ``src/cpp`` via scikit-build-core. When it is not
available (no compiler on the machine, or ``NESYDEP_SKIP_CPP=1``), importing
this module still works; instantiating a native miner raises a clear error.

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
    func_name: str,
    subtables: list[SubTable],
    config_dict: dict[str, Any],
    n_jobs: int,
) -> list[dict]:
    core = _core()
    func = getattr(core, func_name)
    payloads = [_frame_payload(st) for st in subtables]
    if n_jobs <= 1 or len(payloads) <= 1:
        results = [func(cols, rows, config_dict) for cols, rows in payloads]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_call_native, [(func_name, c, r, config_dict) for c, r in payloads]))
    return [dep for batch in results for dep in batch]


def _call_native(args: tuple[str, list[str], list[list[str]], dict]) -> list[dict]:
    func_name, cols, rows, config = args
    return getattr(_core(), func_name)(cols, rows, config)


def _n_jobs(config: Any) -> int:
    return int(getattr(config, "n_jobs", 1))


class _NativeFDMinerBase:
    func_name = ""

    def mine(self, subtables: list[SubTable], config: FDMinerConfig | None = None) -> list[FD]:
        cfg = config or FDMinerConfig()
        cfg_dict = cfg.model_dump()
        raw = _run_parallel(self.func_name, subtables, cfg_dict, _n_jobs(cfg))
        return [FD(lhs=tuple(d["lhs"]), rhs=d["rhs"]) for d in raw]


@MINERS.decorator("pfminer")
class PFMinerNative(_NativeFDMinerBase):
    """BSFD's parallel FD kernel (native)."""

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
        raw = _run_parallel("cfd_mine", subtables, cfg.model_dump(), _n_jobs(cfg))
        return [
            CFD(
                lhs=tuple(d["lhs"]),
                rhs=d["rhs"],
                lhs_pattern=tuple(d["lhs_pattern"]),
                rhs_pattern=d["rhs_pattern"],
            )
            for d in raw
        ]


@MINERS.decorator("ctane")
class CTaneNative(SCFDMNative):
    """CTane baseline — SCFDM's 'Integrated-BFS' strategy."""

    def mine(
        self, subtables: list[SubTable], config: CFDMinerConfig | None = None
    ) -> list[Dependency]:
        cfg = (config or CFDMinerConfig()).model_copy(update={"strategy": "Integrated-BFS"})
        return super().mine(subtables, cfg)
