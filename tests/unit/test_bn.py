"""BN SubLearner correlation tests — requires the ``bn`` extra (pgmpy)."""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pgmpy")
pytestmark = pytest.mark.requires_bn

from nesydep.core.config import BNCorrelationConfig  # noqa: E402
from nesydep.core.registry import CORRELATORS  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "toy_addresses.csv"


def test_bn_correlation_toy_triangle():
    """zip/city/state deterministically encode each other; the Markov
    blankets must recover this triangle."""
    frame = pd.read_csv(DATA)
    graph = CORRELATORS.get("bn")().extract(frame, BNCorrelationConfig())
    cols = {frozenset(s.columns) for s in graph.sets}
    # every set involving one of the triangle must contain another member
    triangle = {"zip", "city", "state"}
    triangle_sets = [c for c in cols if c & triangle]
    assert triangle_sets, "no correlated set touches the zip/city/state triangle"
    assert any(len(c & triangle) >= 2 for c in triangle_sets)


def test_bn_skips_constant_columns():
    frame = pd.read_csv(DATA).assign(const="x")
    graph = CORRELATORS.get("bn")().extract(frame, BNCorrelationConfig())
    assert all("const" not in s.columns for s in graph.sets)
