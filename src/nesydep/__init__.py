"""NeSyDep — fast neural-symbolic discovery of data dependencies.

Typical usage::

    import nesydep as nd

    fds = nd.discover(df, algo="bsfd", support=100)
    miner = nd.get_algorithm("scfdm")(correlation="pearson")
    result = miner.discover(df)
    metrics = nd.evaluate(result, ground_truth="goldens/hospital.txt")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Importing the built-in stage/algorithm subpackages registers their members.
from nesydep import algorithms as _algorithms  # noqa: F401
from nesydep import correlation as _correlation  # noqa: F401
from nesydep import miners as _miners  # noqa: F401
from nesydep import partition as _partition  # noqa: F401
from nesydep import sampling as _sampling  # noqa: F401
from nesydep.algorithms import BSFD, SCFDM
from nesydep.core.dependency import CFD, FD, WILDCARD, Dependency
from nesydep.core.pipeline import Pipeline
from nesydep.core.registry import ALGORITHMS
from nesydep.core.results import MiningResult

__version__ = "0.1.1"

__all__ = [
    "BSFD",
    "SCFDM",
    "CFD",
    "FD",
    "WILDCARD",
    "Dependency",
    "MiningResult",
    "Pipeline",
    "discover",
    "evaluate",
    "get_algorithm",
    "list_algorithms",
]


def get_algorithm(name: str) -> type:
    """Return the algorithm class registered under ``name``."""
    return ALGORITHMS.get(name)


def list_algorithms() -> list[str]:
    """Names of all registered algorithms (built-in + plugins)."""
    return ALGORITHMS.names()


def discover(data: Any, algo: str = "bsfd", **params: Any) -> MiningResult:
    """One-call dependency discovery.

    Args:
        data: a pandas DataFrame, a path to a CSV/parquet file, or a Dataset.
        algo: registered algorithm name (see :func:`list_algorithms`).
        **params: algorithm parameters, e.g. ``support=100, confidence=0.95``.

    Returns:
        A :class:`MiningResult` with the discovered dependencies.
    """
    algorithm = ALGORITHMS.get(algo)
    return algorithm(**params).discover(data)


def evaluate(result: Any, ground_truth: Any) -> Any:
    """P/R/F1 of a result (or dependency list) against ground truth.

    ``ground_truth`` may be a list of dependencies or a path to a legacy
    text file (``[A, B] -> C`` / ``[A, B] => C, (a || b)`` lines).
    """
    from nesydep.evaluation.metrics import evaluate as _evaluate
    from nesydep.io.legacy import load_dependencies

    mined = result.dependencies if isinstance(result, MiningResult) else list(result)
    if isinstance(ground_truth, (str, Path)):
        truth = load_dependencies(ground_truth)
    else:
        truth = list(ground_truth)
    return _evaluate(mined, truth)
