"""Sampler and correlation-extractor unit tests."""

from pathlib import Path

import pandas as pd
import pytest

import nesydep as nd  # noqa: F401  (registers built-ins)
from nesydep.core.config import RandomSampleConfig, StratifiedSampleConfig
from nesydep.core.registry import CORRELATORS, SAMPLERS

DATA = Path(__file__).resolve().parents[1] / "data" / "toy_addresses.csv"


@pytest.fixture(scope="module")
def frame():
    return pd.read_csv(DATA)


def test_random_sampler_size(frame):
    sampler = SAMPLERS.get("random")()
    out = sampler.sample(frame, RandomSampleConfig(ratio=0.5, seed=1))
    assert len(out) == 6


def test_stratified_sampler_covers_strata(frame):
    sampler = SAMPLERS.get("stratified")()
    out = sampler.sample(frame, StratifiedSampleConfig(ratio=0.4, stratify_by="city", seed=1))
    assert set(out["city"]) == set(frame["city"])  # every stratum survives
    assert 0 < len(out) < len(frame)


def test_lightweight_correlation_finds_zip_city(frame):
    extractor = CORRELATORS.get("lightweight")()
    graph = extractor.extract(frame)
    # zip fully determines city: their codes are perfectly associated, so
    # "city" must appear in some correlated set together with "zip".
    cols = [set(s.columns) for s in graph.sets]
    assert any({"zip", "city"} <= c for c in cols)


def test_repsampler_import_guard():
    pytest.importorskip("datasketch")
    from nesydep.core.config import RepresentativeSampleConfig

    sampler = SAMPLERS.get("representative")()
    frame = pd.read_csv(DATA)
    out = sampler.sample(frame, RepresentativeSampleConfig(bound=6, seed=1))
    assert len(out) == 6


def test_sampling_bounds():
    from nesydep.sampling.bounds import (
        compare_bounds,
        p_hat_estimate,
        theorem2_bound,
    )

    bounds = compare_bounds(p_hat=0.01, eps=5e-4, eta=0.9, lam=0.05)
    assert set(bounds) == {
        "variance_augmented",
        "chebyshev",
        "bennett",
        "hybrid_chernoff",
        "theorem2",
    }
    assert all(v >= 1 for v in bounds.values())
    frame = pd.read_csv(DATA)
    p_hat = p_hat_estimate(frame, delta_hat=2e-3)
    assert 0 < p_hat <= 0.5
    assert theorem2_bound(p_hat, 5e-4, 0.9, 0.05) >= 1
