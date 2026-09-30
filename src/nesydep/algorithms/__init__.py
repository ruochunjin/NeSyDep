"""Algorithm presets: named, ready-to-run pipeline compositions."""
from __future__ import annotations

import warnings
from typing import Any

from nesydep.core.config import (
    BSFDConfig,
    CFDMinerConfig,
    FDMinerConfig,
    LightweightCorrelationConfig,
    RepresentativeSampleConfig,
    SCFDMConfig,
    VerticalPartitionConfig,
)
from nesydep.core.pipeline import Pipeline
from nesydep.core.registry import ALGORITHMS


class MiningAlgorithm:
    """Base class: an algorithm is a preset pipeline plus its config."""

    name = "abstract"

    def __init__(self, config: Any = None, **overrides: Any) -> None:
        self.config = config if config is not None else self._default_config()
        for key, value in overrides.items():
            self._apply_override(key, value)

    def _default_config(self) -> Any:
        raise NotImplementedError

    def _apply_override(self, key: str, value: Any) -> None:
        # Convenience: ``BSFD(support=100)`` sets ``config.miner.support``.
        for section in ("miner", "sampler", "correlation", "global_"):
            sub = getattr(self.config, section, None)
            if sub is not None and hasattr(sub, key):
                setattr(sub, key, value)
                return
        raise TypeError(f"unknown parameter {key!r} for algorithm {self.name!r}")

    def _pipeline(self) -> Pipeline:
        raise NotImplementedError

    def discover(self, data: Any) -> Any:
        return self._pipeline().run(data)


@ALGORITHMS.decorator("bsfd")
class BSFD(MiningAlgorithm):
    """Fast FD discovery via Bayesian-network learning (ICDE 2026)."""

    name = "bsfd"

    def _default_config(self) -> BSFDConfig:
        return BSFDConfig()

    def _pipeline(self) -> Pipeline:
        cfg: BSFDConfig = self.config
        return Pipeline(
            sampler="stratified",
            correlation="bn",
            partitioner="vertical",
            miner="pfminer",
            algorithm=self.name,
            configs={
                "sampler": cfg.sampler,
                "correlation": cfg.correlation,
                "partitioner": VerticalPartitionConfig(),
                "miner": cfg.miner,
            },
        )


@ALGORITHMS.decorator("scfdm")
class SCFDM(MiningAlgorithm):
    """Fast CFD discovery via Transformer-guided relation partitioning.

    ``correlation="transformer"`` requires the ``transformer`` extra (torch);
    without it the pipeline automatically falls back to lightweight
    statistical correlation with a warning.
    """

    name = "scfdm"

    def _default_config(self) -> SCFDMConfig:
        return SCFDMConfig()

    def _pipeline(self) -> Pipeline:
        cfg: SCFDMConfig = self.config
        correlation = cfg.correlation
        if getattr(correlation, "name", None) == "transformer":
            try:
                import torch  # noqa: F401
            except ImportError:
                warnings.warn(
                    "correlation='transformer' requires the 'transformer' extra "
                    "(pip install 'nesydep[transformer]'); falling back to "
                    "lightweight Pearson correlation.",
                    stacklevel=2,
                )
                correlation = LightweightCorrelationConfig()
        return Pipeline(
            sampler="representative",
            correlation=correlation.name,
            partitioner="vertical",
            miner="scfdm",
            algorithm=self.name,
            configs={
                "sampler": cfg.sampler
                if isinstance(cfg.sampler, RepresentativeSampleConfig)
                else RepresentativeSampleConfig(),
                "correlation": correlation,
                "partitioner": VerticalPartitionConfig(),
                "miner": cfg.miner
                if isinstance(cfg.miner, CFDMinerConfig)
                else CFDMinerConfig(),
            },
        )


def _baseline(algo_name: str, miner_name: str, miner_config: Any) -> type[MiningAlgorithm]:
    """End-to-end baselines = identity stages + a single miner."""

    class Baseline(MiningAlgorithm):
        name = algo_name

        def _default_config(self) -> Any:
            return miner_config.model_copy()

        def _apply_override(self, key: str, value: Any) -> None:
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise TypeError(f"unknown parameter {key!r} for algorithm {algo_name!r}")

        def _pipeline(self) -> Pipeline:
            return Pipeline(
                miner=miner_name, algorithm=algo_name, configs={"miner": self.config}
            )

    Baseline.__name__ = algo_name.upper().replace("-", "_")
    return ALGORITHMS.register(algo_name, Baseline)


_baseline("tane", "tane", FDMinerConfig(name="tane"))
_baseline("dfd", "dfd", FDMinerConfig(name="dfd"))
_baseline("ctane", "ctane", CFDMinerConfig())
_baseline("pyref-fd", "pyref-fd", FDMinerConfig(name="pfminer"))
