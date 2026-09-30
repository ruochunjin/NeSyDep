"""Configuration: pydantic v2 models, layered per stage.

Every tunable of the prototypes (including the "magic numbers" that lived in
scripts) becomes a typed field here; defaults carry the paper values.
``"auto"`` sentinel values trigger the papers' adaptive mechanisms
(sampling lower bound, entropy-based thresholds) instead of fixed numbers.
"""
from __future__ import annotations

import os
from typing import Literal, Union

from pydantic import BaseModel, Field


class GlobalConfig(BaseModel):
    n_jobs: int = Field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 1))
    seed: int = 42
    openmp_threads: int = Field(default_factory=lambda: min(8, os.cpu_count() or 1))
    tmp_dir: str | None = None  # None -> system temp


class StageConfig(BaseModel):
    """Base for per-stage configs. ``name`` selects the registered impl."""

    name: str


# -- sampling ------------------------------------------------------------------


class StratifiedSampleConfig(StageConfig):
    """BSFD's stratified sampling."""

    name: Literal["stratified"] = "stratified"
    ratio: float = Field(default=0.15, gt=0, le=1.0)
    stratify_by: str | None = None  # None -> pick a high-cardinality column
    seed: int = 42


class RepresentativeSampleConfig(StageConfig):
    """SCFDM's RepSampler (MinHash-LSH, diversity-first).

    ``bound="auto"`` computes the paper's Theorem-1/2 minimum sample size
    from the data instead of using a fixed count.
    """

    name: Literal["representative"] = "representative"
    bound: Union[int, Literal["auto"]] = "auto"
    num_perm: int = 128
    seed: int = 42
    eps: float = 5e-4  # sampling error, used when bound == "auto"
    eta: float = 0.9  # confidence, used when bound == "auto"
    lam: float = 0.05  # Chernoff ratio, used when bound == "auto"


class RandomSampleConfig(StageConfig):
    name: Literal["random"] = "random"
    ratio: float = Field(default=0.15, gt=0, le=1.0)
    seed: int = 42


SamplerConfig = Union[StratifiedSampleConfig, RepresentativeSampleConfig, RandomSampleConfig]

# -- correlation extraction ------------------------------------------------------


class BNCorrelationConfig(StageConfig):
    """BSFD's SubLearner: per-node Bayesian-network structure learning."""

    name: Literal["bn"] = "bn"
    score: str = "aic"
    max_parents: int = 5


class TransformerCorrelationConfig(StageConfig):
    """SCFDM's AttrFinder (requires the ``transformer`` extra).

    ``threshold="auto"`` enables the paper's normalised-Shannon-entropy
    adaptive retention threshold.
    """

    name: Literal["transformer"] = "transformer"
    threshold: Union[float, Literal["auto"]] = "auto"
    gamma0: float = 0.85  # retention density floor (paper default)
    z: int = 3  # significance threshold (paper default)
    epochs: int = 12
    d_model: int = 64
    device: str = "auto"  # "auto" -> cuda if available else cpu


class LightweightCorrelationConfig(StageConfig):
    """Statistical correlation (Pearson/PCA/Lasso) — no GPU needed."""

    name: Literal["lightweight"] = "lightweight"
    method: Literal["pearson", "pca", "lasso"] = "pearson"
    threshold: float = 0.3
    max_antecedents: int = 10


CorrelationConfig = Union[
    BNCorrelationConfig, TransformerCorrelationConfig, LightweightCorrelationConfig
]

# -- partitioning -----------------------------------------------------------------


class VerticalPartitionConfig(StageConfig):
    name: Literal["vertical"] = "vertical"
    max_columns_per_table: int | None = None  # None -> no cap


# -- mining ------------------------------------------------------------------------


class FDMinerConfig(StageConfig):
    """Config for C++ FD mining kernels (pfminer / tane / dfd)."""

    name: Literal["pfminer", "tane", "dfd"] = "pfminer"
    support: int = 1  # absolute minimum support (rows)
    confidence: float = Field(default=1.0, gt=0, le=1.0)
    max_lhs: int = Field(default=0, ge=0)  # 0 -> unlimited


class CFDMinerConfig(StageConfig):
    """Config for the C++ CFD kernel (scfdm); strategy names match the paper."""

    name: Literal["scfdm"] = "scfdm"
    support: int = 1
    confidence: float = Field(default=1.0, gt=0, le=1.0)
    max_lhs: int = Field(default=3, ge=1)
    strategy: str = "FD-First-DFS-dfs"  # fastest in the paper's experiments
    constant_only: bool = False  # True -> SCFDM_part behaviour


MinerConfig = Union[FDMinerConfig, CFDMinerConfig]


# -- algorithm presets --------------------------------------------------------------


class BSFDConfig(BaseModel):
    """End-to-end config for the BSFD algorithm (FD discovery)."""

    global_: GlobalConfig = Field(default_factory=GlobalConfig)
    sampler: StratifiedSampleConfig = Field(default_factory=StratifiedSampleConfig)
    correlation: BNCorrelationConfig = Field(default_factory=BNCorrelationConfig)
    miner: FDMinerConfig = Field(default_factory=FDMinerConfig)


class SCFDMConfig(BaseModel):
    """End-to-end config for the SCFDM algorithm (CFD discovery)."""

    global_: GlobalConfig = Field(default_factory=GlobalConfig)
    sampler: RepresentativeSampleConfig = Field(default_factory=RepresentativeSampleConfig)
    correlation: Union[TransformerCorrelationConfig, LightweightCorrelationConfig] = Field(
        default_factory=LightweightCorrelationConfig
    )
    miner: CFDMinerConfig = Field(default_factory=CFDMinerConfig)
