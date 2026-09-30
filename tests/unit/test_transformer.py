"""AttrFinder (Transformer) correlation extraction tests — requires torch."""

from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.requires_torch

from nesydep.core.config import TransformerCorrelationConfig  # noqa: E402
from nesydep.core.registry import CORRELATORS  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "toy_addresses.csv"


def test_transformer_correlation_on_toy():
    frame = pd.read_csv(DATA)
    extractor = CORRELATORS.get("transformer")()
    # tiny model + few epochs: we only assert structure, not quality
    cfg = TransformerCorrelationConfig(epochs=2, d_model=16, threshold=0.0)
    graph = extractor.extract(frame, cfg)
    # threshold 0.0 keeps every non-self pair with any positive importance;
    # at minimum the deterministic zip->city/state structure should yield
    # some correlated sets, and every set must reference real columns.
    assert graph.sets, "expected at least one correlated set"
    cols = set(frame.columns)
    for s in graph.sets:
        assert s.target in cols
        assert set(s.antecedents) <= cols
        assert s.target not in s.antecedents
