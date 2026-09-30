"""End-to-end pipeline tests on the toy dataset (no C++ toolchain needed).

The naive reference miner is exact, so on ``toy_addresses.csv`` it must
recover the hand-verified ground truth.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from toy_ground_truth import ALL_MINIMAL_FDS  # noqa: E402

import nesydep as nd  # noqa: E402
from nesydep.core.dependency import CFD, FD  # noqa: E402
from nesydep.evaluation.verify import verify  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "toy_addresses.csv"


@pytest.fixture(scope="module")
def frame():
    return pd.read_csv(DATA)


def test_naive_miner_finds_ground_truth(frame):
    result = nd.Pipeline(miner="pyref-fd", algorithm="pyref-fd").run(frame)
    found = {(fd.lhs, fd.rhs) for fd in result.fds}
    # The naive miner is exact, so on a 12-row table it also finds incidental
    # FDs (any large-enough column combo is unique and hence a determinant).
    # We therefore check recall against the hand-verified set, and soundness
    # of everything it reports.
    assert ALL_MINIMAL_FDS <= found
    assert all(v.holds for v in verify(frame.astype("string"), result.fds))


def test_discover_one_liner_with_reference_miner(frame):
    result = nd.discover(frame, algo="pyref-fd")
    assert ALL_MINIMAL_FDS <= {(fd.lhs, fd.rhs) for fd in result.fds}
    assert result.algorithm == "pyref-fd"
    assert result.stats["n_rows"] == 12


def test_pipeline_stats_and_snapshot(frame):
    result = nd.Pipeline(miner="pyref-fd", algorithm="test").run(frame)
    assert result.stats["n_subtables"] == 1
    assert "mine_seconds" in result.stats


def test_result_roundtrip_json(frame, tmp_path):
    result = nd.discover(frame, algo="pyref-fd")
    out = tmp_path / "result.json"
    result.to_json(out)
    loaded = nd.MiningResult.from_json(out)
    assert set(loaded.dependencies) == set(result.dependencies)


def test_verify_constant_cfd(frame):
    frame_str = frame.astype("string")
    cfd = CFD(("city",), "state", ("New York",), "NY")
    (v,) = verify(frame_str, [cfd])
    assert v.holds and v.support == 4 and v.confidence == 1.0


def test_verify_variable_cfd(frame):
    frame_str = frame.astype("string")
    cfd = CFD(("zip",), "city", ("_",), "_")
    (v,) = verify(frame_str, [cfd])
    assert v.holds and v.confidence == 1.0


def test_verify_detects_violation(frame):
    frame_str = frame.astype("string")
    cfd = CFD(("city",), "state", ("New York",), "CA")  # wrong state
    (v,) = verify(frame_str, [cfd])
    assert not v.holds and v.support == 0


def test_evaluate_metrics(frame):
    result = nd.discover(frame, algo="pyref-fd")
    metrics = nd.evaluate(result, [FD(lhs, rhs) for lhs, rhs in ALL_MINIMAL_FDS])
    # recall 1.0: every intended FD is found; precision < 1 due to incidental FDs
    assert metrics.recall == 1.0 and metrics.precision > 0
