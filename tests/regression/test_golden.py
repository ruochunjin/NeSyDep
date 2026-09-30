"""Regression tests against frozen golden outputs.

Every golden was produced by ``freeze_golden.py`` only after independent
cross-checks passed (TANE textbook-verified holds+minimal, DFD ⊆ TANE,
within-family strategy agreement, CFD semantic verification). See
``tests/regression/README_golden.md`` for provenance.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_core = pytest.importorskip("nesydep._core")
pytestmark = pytest.mark.requires_cpp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
GOLDEN = Path(__file__).resolve().parent / "golden"
DATA = ROOT / "tests" / "data"

MANIFEST = json.loads((GOLDEN / "manifest.json").read_text(encoding="utf-8"))


def load_golden_fd(filename: str) -> set:
    from nesydep.io.legacy import load_dependencies

    return {(d.lhs, d.rhs) for d in load_dependencies(GOLDEN / filename)}


def load_golden_cfd(filename: str) -> set:
    from nesydep.io.legacy import load_dependencies

    return {
        (tuple(sorted(zip(d.lhs, d.lhs_pattern, strict=True))), d.rhs, d.rhs_pattern)
        for d in load_dependencies(GOLDEN / filename)
    }


def payload(name: str, col_prefix: int | None = None):
    frame = pd.read_csv(DATA / name)
    if col_prefix:
        frame = frame.iloc[:, :col_prefix]
    return [str(c) for c in frame.columns], frame.astype(str).values.tolist()


@pytest.mark.parametrize(
    "tag,csv_file",
    [("toy", "toy_addresses.csv"), ("german16", "german_credit_sample.csv")],
)
def test_fulltable_fd_golden(tag, csv_file):
    key = "toy_fulltable" if tag == "toy" else "german16_fulltable"
    params = MANIFEST[key]["params"]
    cols, rows = payload(csv_file, params["col_prefix"])
    mined = {
        (tuple(sorted(lhs)), r)
        for lhs, r in _core.tane_mine(
            cols, rows, params["support"], params["confidence"], params["max_lhs"]
        )
    }
    assert mined == load_golden_fd(MANIFEST[key]["file"])


def test_pipeline_fd_golden_reproduces():
    from nesydep.core.config import (
        FDMinerConfig,
        LightweightCorrelationConfig,
        StratifiedSampleConfig,
        VerticalPartitionConfig,
    )
    from nesydep.core.pipeline import Pipeline

    for key, csv_file in [
        ("german_pipeline", "german_credit_sample.csv"),
        ("census_pipeline", "census42_sample.csv"),
    ]:
        pipe = Pipeline(
            sampler="stratified",
            correlation="lightweight",
            partitioner="vertical",
            miner="pfminer",
            algorithm=f"golden-{key}",
            configs={
                "sampler": StratifiedSampleConfig(ratio=0.5, seed=7),
                "correlation": LightweightCorrelationConfig(method="pearson", threshold=0.2),
                "partitioner": VerticalPartitionConfig(),
                "miner": FDMinerConfig(support=5, confidence=0.95),
            },
        )
        result = pipe.run(pd.read_csv(DATA / csv_file))
        assert {(fd.lhs, fd.rhs) for fd in result.fds} == load_golden_fd(MANIFEST[key]["file"]), (
            f"pipeline golden mismatch on {key}"
        )


@pytest.mark.parametrize("family", ["fd-first", "ctane"])
@pytest.mark.parametrize(
    "tag,csv_file",
    [
        ("toy", "toy_addresses.csv"),
        ("german", "german_credit_sample.csv"),
        ("census42", "census42_sample.csv"),
    ],
)
def test_cfd_golden(family, tag, csv_file):
    key = {"toy": "toy_cfd", "german": "german_cfd", "census42": "census_cfd"}[tag]
    params = MANIFEST[key]["params"]
    cols, rows = payload(csv_file)
    strategy = "FD-First-DFS-dfs" if family == "fd-first" else "Integrated-BFS"
    mined = {
        (tuple(sorted(zip(lhs_attrs, patterns, strict=True))), r, rp)
        for lhs_attrs, r, patterns, rp in _core.cfd_mine(
            cols, rows, params["support"], params["confidence"], params["max_lhs"], strategy, False
        )
        if lhs_attrs  # exclude trivial empty-LHS CFDs (documented kernel quirk)
    }
    assert mined == load_golden_cfd(MANIFEST[key][family]["file"])
