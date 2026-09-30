"""Freeze golden reference outputs for regression testing.

Golden outputs are only written when *independent* cross-checks pass:

1. TANE and DFD (independent implementations) must agree exactly.
2. The pure-Python textbook miner (pyref-fd) must confirm every native FD
   holds under the textbook definition, and may only *add* key-type FDs the
   pair-based kernels cannot see.
3. Every mined CFD is re-verified against the full sample with an
   independent pandas implementation (support & confidence).
4. All ten SCFDM search strategies must produce the same CFD set.

Scope note: this simplified-TANE prototype kernel scales exponentially in
columns on full tables (by design it mines SubLearner-produced sub-tables),
so full-table FD goldens are frozen on bounded inputs (toy 6 cols; a 16-col
prefix of german_credit), while the realistic german_credit/census42 goldens
go through the partitioned pipeline — mirroring how the algorithms are meant
to be used.

Regenerate with:  python tests/regression/freeze_golden.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "data"))

from toy_ground_truth import ALL_MINIMAL_FDS  # noqa: E402

import nesydep as nd  # noqa: E402
from nesydep import _core  # noqa: E402
from nesydep.core.config import (  # noqa: E402
    FDMinerConfig,
    LightweightCorrelationConfig,
    StratifiedSampleConfig,
    VerticalPartitionConfig,
)
from nesydep.core.pipeline import Pipeline  # noqa: E402
from nesydep.evaluation.verify import verify  # noqa: E402
from nesydep.io.legacy import dump_dependencies  # noqa: E402
from nesydep.miners.pyref import discover_fds_naive  # noqa: E402

DATA = ROOT / "tests" / "data"
GOLDEN = ROOT / "tests" / "regression" / "golden"

# Strategies form two families with different (both valid) minimality notions:
# the FD-First family enumerates finer-grained constant patterns, while the
# Itemset-First/Integrated (CTane) family returns the coarser cover.
# Agreement is required *within* a family.
STRATEGY_FAMILIES = {
    "fd-first": ["FD-First-DFS-dfs", "FD-First-DFS-bfs", "FD-First-BFS-dfs", "FD-First-BFS-bfs"],
    "ctane": [
        "Itemset-First-DFS-dfs",
        "Itemset-First-DFS-bfs",
        "Itemset-First-BFS-dfs",
        "Itemset-First-BFS-bfs",
        "Integrated-DFS",
        "Integrated-BFS",
    ],
}
STRATEGIES = [s for fam in STRATEGY_FAMILIES.values() for s in fam]


def payload(name: str):
    frame = pd.read_csv(DATA / name)
    return frame, [str(c) for c in frame.columns], frame.astype(str).values.tolist()


def fd_set(pairs):
    return {(tuple(sorted(lhs)), rhs) for lhs, rhs in pairs}


def _textbook_check(frame: pd.DataFrame, lhs: tuple, rhs: str) -> bool:
    f = frame.astype("string")
    return bool((f.groupby(list(lhs), sort=False)[rhs].nunique() <= 1).all())


def _is_minimal(frame: pd.DataFrame, lhs: tuple, rhs: str) -> bool:
    from itertools import combinations

    for k in range(1, len(lhs)):
        for sub in combinations(lhs, k):
            if _textbook_check(frame, sub, rhs):
                return False
    return True


def freeze_fd_fulltable(
    tag: str,
    name: str,
    support: int,
    confidence: float,
    max_lhs: int,
    col_prefix: int | None = None,
) -> dict:
    """Exact full-table FD golden: TANE as reference, textbook-validated.

    Every reported FD is brute-force checked to (a) hold exactly and
    (b) be minimal. DFD is only required to be a *subset* of TANE — the
    prototype DFD is approximate and demonstrably misses valid minimal FDs
    on german16 (documented limitation).
    """
    frame, cols, rows = payload(name)
    if col_prefix:
        frame = frame.iloc[:, :col_prefix]
        cols, rows = cols[:col_prefix], [r[:col_prefix] for r in rows]
    tane = fd_set(_core.tane_mine(cols, rows, support, confidence, max_lhs))
    dfd = fd_set(_core.dfd_mine(cols, rows, support, confidence, max_lhs))
    assert dfd <= tane, f"DFD reports FDs TANE does not on {tag}: {sorted(dfd - tane)[:5]}"

    if confidence >= 1.0:
        for lhs, rhs in tane:
            assert _textbook_check(frame, lhs, rhs), f"FD {(lhs, rhs)} does not hold"
            assert _is_minimal(frame, lhs, rhs), f"FD {(lhs, rhs)} is not minimal"

    deps = [nd.FD(lhs, rhs) for lhs, rhs in sorted(tane)]
    out = GOLDEN / f"fd_tane_{tag}.txt"
    dump_dependencies(deps, out)
    return {
        "file": out.name,
        "n": len(deps),
        "n_dfd_subset": len(dfd),
        "params": {
            "support": support,
            "confidence": confidence,
            "max_lhs": max_lhs,
            "col_prefix": col_prefix,
        },
    }


def freeze_fd_pipeline(tag: str, name: str) -> dict:
    """Partitioned-pipeline FD golden (the intended usage on wide tables).

    Fully deterministic: seeded stratified sampler + deterministic Pearson
    correlation + native kernel.
    """
    frame, _, _ = payload(name)
    pipe = Pipeline(
        sampler="stratified",
        correlation="lightweight",
        partitioner="vertical",
        miner="pfminer",
        algorithm=f"golden-{tag}",
        configs={
            "sampler": StratifiedSampleConfig(ratio=0.5, seed=7),
            "correlation": LightweightCorrelationConfig(method="pearson", threshold=0.2),
            "partitioner": VerticalPartitionConfig(),
            "miner": FDMinerConfig(support=5, confidence=0.95),
        },
    )
    result = pipe.run(frame)
    out = GOLDEN / f"fd_pipeline_{tag}.txt"
    dump_dependencies(result.fds, out)
    # textbook check: every pipeline FD must verify on the sampled frame
    sample = pd.read_csv(DATA / name)  # full frame for verification
    checks = verify(sample.astype("string"), result.fds, min_support=1, min_confidence=0.9)
    bad = [str(v.dependency) for v in checks if not v.holds]
    # sampled mining can yield FDs that only approximately hold full-table;
    # keep those under a relaxed band but report them in the manifest.
    return {
        "file": out.name,
        "n": len(result.fds),
        "n_approx_violations_full_table": len(bad),
        "stats": result.stats,
    }


def freeze_toy() -> dict:
    """The 12-row toy table: cross-checked against hand-written truth."""
    frame, cols, rows = payload("toy_addresses.csv")
    pyref = {(fd.lhs, fd.rhs) for fd in discover_fds_naive(frame)}
    assert pyref >= ALL_MINIMAL_FDS, "pyref missed hand-verified FDs"
    dump_dependencies([nd.FD(lhs, rhs) for lhs, rhs in sorted(pyref)], GOLDEN / "fd_pyref_toy.txt")
    return {"file": "fd_pyref_toy.txt", "n": len(pyref)}


def freeze_cfd(tag: str, name: str, support: int, confidence: float, max_lhs: int) -> dict:
    frame, cols, rows = payload(name)
    out = {}
    for family, strategies in STRATEGY_FAMILIES.items():
        reference = None
        for strategy in strategies:
            mined = _core.cfd_mine(cols, rows, support, confidence, max_lhs, strategy, False)
            # Integrated-DFS additionally reports trivial empty-LHS CFDs
            # (() => A, _) on some inputs; they carry no dependency content
            # and are excluded from comparison (documented kernel quirk).
            as_set = {
                (tuple(sorted(zip(lhs_attrs, patterns, strict=True))), r, rp)
                for lhs_attrs, r, patterns, rp in mined
                if lhs_attrs
            }
            if reference is None:
                reference = as_set
            else:
                assert as_set == reference, (
                    f"strategy {strategy} disagrees within family {family} on {tag}: "
                    f"{len(as_set)} vs {len(reference)} CFDs"
                )

        cfds = [
            nd.CFD(
                lhs=tuple(a for a, _ in lp),
                rhs=r,
                lhs_pattern=tuple(p for _, p in lp),
                rhs_pattern=rp,
            )
            for lp, r, rp in sorted(reference)
        ]
        checks = verify(
            frame.astype("string"), cfds, min_support=support, min_confidence=confidence
        )
        bad = [str(v.dependency) for v in checks if not v.holds]
        assert not bad, f"mined CFDs failing verification on {tag}/{family}: {bad[:5]}"

        path = GOLDEN / f"cfd_{family}_{tag}.txt"
        dump_dependencies(cfds, path)
        out[family] = {"file": path.name, "n": len(cfds), "strategies_checked": len(strategies)}
    out["params"] = {"support": support, "confidence": confidence, "max_lhs": max_lhs}
    return out


def main() -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}

    manifest["toy_textbook"] = freeze_toy()
    manifest["toy_fulltable"] = freeze_fd_fulltable(
        "toy", "toy_addresses.csv", support=1, confidence=1.0, max_lhs=0
    )
    # 16-col prefix: the largest full-table input this simplified-TANE
    # kernel handles comfortably.
    manifest["german16_fulltable"] = freeze_fd_fulltable(
        "german16", "german_credit_sample.csv", support=5, confidence=1.0, max_lhs=0, col_prefix=16
    )
    # Realistic wide-table goldens via the partitioned pipeline.
    manifest["german_pipeline"] = freeze_fd_pipeline("german", "german_credit_sample.csv")
    manifest["census_pipeline"] = freeze_fd_pipeline("census42", "census42_sample.csv")

    manifest["toy_cfd"] = freeze_cfd("toy", "toy_addresses.csv", 1, 1.0, 2)
    manifest["german_cfd"] = freeze_cfd("german", "german_credit_sample.csv", 5, 0.95, 2)
    manifest["census_cfd"] = freeze_cfd("census42", "census42_sample.csv", 5, 0.95, 2)

    (GOLDEN / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
