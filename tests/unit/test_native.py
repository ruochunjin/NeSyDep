"""Native C++ kernel tests — skipped when the extension is unavailable.

These double as the first regression layer: TANE and DFD are independent
implementations and must agree on exact FD discovery; CFD results are
cross-checked against `nesydep.evaluation.verify`.

Note on semantics: the BSFD prototype's TANE/DFD use pair-based support and
confidence (sup / (n²−n)). FDs whose LHS groups are all singletons — e.g.
key FDs — are invisible to that measure (0/0 confidence), so the toy key
column's FDs are not expected from the native kernels. The pure-Python
reference miner uses the textbook definition and does find them.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

_core = pytest.importorskip("nesydep._core")
pytestmark = pytest.mark.requires_cpp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from toy_ground_truth import MINIMAL_FDS  # noqa: E402  (non-key FDs)

DATA = Path(__file__).resolve().parents[1] / "data" / "toy_addresses.csv"


@pytest.fixture(scope="module")
def payload():
    frame = pd.read_csv(DATA)
    return [str(c) for c in frame.columns], frame.astype(str).values.tolist()


def test_tane_finds_nonkey_ground_truth(payload):
    cols, rows = payload
    found = {(tuple(sorted(lhs)), r) for lhs, r in _core.tane_mine(cols, rows, 1, 1.0, 0)}
    assert found == MINIMAL_FDS | {
        (("city",), "zip"),
        (("state",), "zip"),
        (("city",), "state"),
        (("state",), "city"),
        (("zip",), "city"),
        (("zip",), "state"),
    }


def test_tane_and_dfd_agree(payload):
    cols, rows = payload
    tane = {(tuple(sorted(lhs)), r) for lhs, r in _core.tane_mine(cols, rows, 1, 1.0, 0)}
    dfd = {(tuple(sorted(lhs)), r) for lhs, r in _core.dfd_mine(cols, rows, 1, 0.95, 0)}
    assert tane == dfd  # independent implementations must agree on exact mining


def test_pfminer_matches_tane(payload):
    cols, rows = payload
    assert _core.pfminer_mine(cols, rows, 1, 1.0, 0) == _core.tane_mine(cols, rows, 1, 1.0, 0)


def test_cfd_mining_constant_and_variable(payload):
    cols, rows = payload
    cfds = _core.cfd_mine(
        cols,
        rows,
        support=1,
        confidence=1.0,
        max_lhs=2,
        strategy="FD-First-DFS-dfs",
        constant_only=False,
    )
    assert len(cfds) > 0
    # spot check: the exact city->state constant CFDs must be present
    tuples = {(tuple(lhs), r, tuple(p), rp) for lhs, r, p, rp in cfds}
    assert (("city",), "state", ("New York",), "NY") in tuples
    # constant_only removes every wildcarded CFD
    const = _core.cfd_mine(cols, rows, 1, 1.0, 2, "FD-First-DFS-dfs", True)
    assert const and all("_" not in pats and rp != "_" for _, _, pats, rp in const)
    assert len(const) < len(cfds)


def test_ctane_strategy(payload):
    cols, rows = payload
    cfds = _core.cfd_mine(cols, rows, 1, 1.0, 2, "Integrated-BFS", False)
    assert len(cfds) > 0


def test_native_miner_through_python_api():
    """The registered 'tane' miner drives the native kernel end to end."""
    import nesydep as nd

    result = nd.discover(pd.read_csv(DATA), algo="tane", support=1, confidence=1.0)
    assert {("city", "state"), ("zip", "city")} <= {
        (fd.lhs[0], fd.rhs) for fd in result.fds if len(fd.lhs) == 1
    }
