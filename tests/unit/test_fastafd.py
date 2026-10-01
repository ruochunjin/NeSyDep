"""FastAFD (FDM, ICDE 2024) integration tests."""

from pathlib import Path

import pandas as pd
import pytest

_core = pytest.importorskip("nesydep._core")
pytestmark = pytest.mark.requires_cpp

import nesydep as nd  # noqa: E402
from nesydep.miners.native import SearchSpaceExplosionError  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "studentfull_26.csv"


@pytest.fixture(scope="module")
def student():
    return pd.read_csv(DATA)


def test_deterministic_per_seed(student):
    a = nd.discover(student, algo="fastafd", seed=42)
    b = nd.discover(student, algo="fastafd", seed=42)
    assert set(a.fds) == set(b.fds) and len(a.fds) > 0


def test_seed_changes_sampled_result(student):
    a = nd.discover(student, algo="fastafd", seed=1)
    b = nd.discover(student, algo="fastafd", seed=99)
    # sampling is the recall mechanism: different seeds sample differently
    assert set(a.fds) != set(b.fds)


def test_golden_student(student):
    """Frozen reference: our deterministic seed=42 output (38 FDs).

    Note: the prototype's own assert (109 FDs matching DFD) only holds for a
    specific libc rand() sequence — the prototype itself produces different
    counts under MinGW (51) and MSVC. FastAFD is approximate by design; the
    golden guards *our* determinism, not cross-platform equality.
    """
    from nesydep.io.legacy import load_dependencies

    golden = load_dependencies(
        Path(__file__).resolve().parents[1] / "regression" / "golden" / "fd_fastafd_student.txt"
    )
    result = nd.discover(student, algo="fastafd", seed=42)
    assert set(result.fds) == set(golden)


def test_column_limit_guard():
    frame = pd.DataFrame({f"c{i}": [i % 3] * 50 for i in range(45)})
    with pytest.raises((SearchSpaceExplosionError, ValueError)):
        nd.discover(frame, algo="fastafd", seed=42)


def test_results_are_plausible_fds(student):
    """Mined FDm must satisfy the paper's pair-based thresholds on full data.

    FDm semantics (ICDE 2024, Def. 3): pair-based error/support. Mining ran
    with error <= 0.03 / support >= 0.2 on the sampled comparison matrix; on
    the full table we allow sampling slack.
    """
    from nesydep.evaluation.verify import verify_fd_approx

    result = nd.discover(student, algo="fastafd", seed=42)
    stats = [verify_fd_approx(student, fd) for fd in result.fds]
    assert all(e <= 0.05 for _, e in stats), max(e for _, e in stats)
    assert all(sup >= 0.15 for sup, _ in stats), min(sup for sup, _ in stats)
