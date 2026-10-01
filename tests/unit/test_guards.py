"""Memory-safety guard tests for the native miners."""
from pathlib import Path

import pandas as pd
import pytest

_core = pytest.importorskip("nesydep._core")
pytestmark = pytest.mark.requires_cpp

from nesydep.core.config import FDMinerConfig  # noqa: E402
from nesydep.miners.native import SearchSpaceExplosionError, TaneNative  # noqa: E402
from nesydep.core.stages import SubTable  # noqa: E402


def test_guard_rejects_wide_full_table():
    frame = pd.DataFrame({f"c{i}": range(100) for i in range(30)})
    with pytest.raises(SearchSpaceExplosionError, match="30 columns"):
        TaneNative().mine([SubTable(frame=frame)], FDMinerConfig())


def test_guard_disabled_with_zero():
    # 12 rows, 6 cols: toy mining must work with the guard off.
    frame = pd.read_csv(Path(__file__).parents[1] / "data" / "toy_addresses.csv")
    out = TaneNative().mine(
        [SubTable(frame=frame)], FDMinerConfig(max_columns_guard=0)
    )
    assert len(out) > 0


def test_guard_passes_narrow_table():
    frame = pd.read_csv(Path(__file__).parents[1] / "data" / "toy_addresses.csv")
    out = TaneNative().mine([SubTable(frame=frame)], FDMinerConfig())
    assert len(out) == 6  # matches the toy golden
