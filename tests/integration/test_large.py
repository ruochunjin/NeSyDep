"""Full-pipeline integration tests on the largest frozen samples.

Marked ``slow`` — excluded from default CI (``pytest -m 'not slow'``), run by
the weekly/manual integration workflow. These exercise the memory-sensitive
paths end to end: wide tables (37–42 columns) must go through the partitioned
pipelines and finish with bounded memory.
"""
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("nesydep._core")
pytestmark = [pytest.mark.slow, pytest.mark.requires_cpp]

import nesydep as nd  # noqa: E402
from nesydep.evaluation.verify import verify  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def alarm():
    return pd.read_csv(DATA / "alarm_sample.csv")  # 500 x 37


@pytest.fixture(scope="module")
def census():
    return pd.read_csv(DATA / "census42_sample.csv")  # 500 x 42


def test_bsfd_wide_table(alarm):
    result = nd.discover(alarm, algo="bsfd", support=10, confidence=0.95)
    assert len(result.fds) > 0
    assert result.stats["n_subtables"] >= 1


def test_scfdm_wide_table(census):
    result = nd.get_algorithm("scfdm")(
        correlation="lightweight", support=10, confidence=0.9, max_lhs=2
    ).discover(census)
    assert len(result.cfds) > 0
    # CFDs mined on a sample are approximate on the full table by design;
    # the paper's guarantee is a high hold rate, not 100%.
    checks = verify(
        census.astype("string"), result.cfds, min_support=10, min_confidence=0.9
    )
    hold_rate = sum(v.holds for v in checks) / len(checks)
    assert hold_rate >= 0.9, f"hold rate {hold_rate:.2f} below 0.9"


def test_scfdm_with_repsampler(census):
    """RepSampler path (MinHash-LSH) on a wide table."""
    pytest.importorskip("datasketch")
    result = nd.get_algorithm("scfdm")(
        sampler="representative", correlation="lightweight",
        support=10, confidence=0.9, max_lhs=2,
    ).discover(census)
    assert len(result.cfds) > 0
    assert result.stats["sample_rows"] <= len(census)


def test_fulltable_tane_guard_fires_wide(census):
    """Full-table TANE on 42 columns must be stopped by the guard, not OOM."""
    from nesydep.miners.native import SearchSpaceExplosionError

    with pytest.raises(SearchSpaceExplosionError):
        nd.discover(census, algo="tane", support=10, confidence=0.95)
