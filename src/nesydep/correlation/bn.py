"""SubLearner: BN-guided correlation extraction (BSFD).

Faithful refactor of the prototype
(``code-for-BSFD/search space reduction/per_node_network_parallel.py``):
for every non-constant target column, learn a Bayesian-network structure
with hill-climbing while *blacklisting all edges not touching the target*,
then take the target's Markov blanket as its correlated attribute set.

Requires the ``bn`` extra (pgmpy). The prototype's ``global_data``
multiprocessing pattern is replaced by plain function arguments.
"""

from __future__ import annotations

import pandas as pd

from nesydep.core.config import BNCorrelationConfig
from nesydep.core.registry import CORRELATORS
from nesydep.core.stages import CorrelatedSet, CorrelationGraph

# pgmpy renamed its scoring methods over releases; map our neutral config
# value to whatever the installed version accepts.
_SCORE_ALIASES = {
    "aic": ("aicscore", "aic-d", "aic_cg"),
    "k2": ("k2score", "k2"),
    "bdeu": ("bdeuscore", "bdeu"),
    "bic": ("bicscore", "bic-d", "bic_cg"),
}


def _learn_markov_blanket(args: tuple) -> list[str]:
    """One BN structure-learning job: Markov blanket of ``node``.

    Supports both pgmpy API generations: the causal-discovery sklearn-style
    API (>= 1.0: ``HillClimbSearch(...).fit(X)`` + ``causal_graph_``) and the
    legacy estimators API (``HillClimbSearch(data).estimate(...)``).
    """
    data, node, score, black_list, max_iter, max_parents = args
    last_err: Exception | None = None
    for candidate in _SCORE_ALIASES.get(score, (score,)):
        try:
            try:
                # pgmpy >= 1.0
                from pgmpy.causal_discovery import (
                    ExpertKnowledge,
                    HillClimbSearch,
                )

                est = HillClimbSearch(
                    scoring_method=candidate,
                    max_indegree=max_parents,
                    max_iter=max_iter,
                    expert_knowledge=ExpertKnowledge(forbidden_edges=black_list),
                    return_type="dag",
                    show_progress=False,
                )
                est.fit(data)
                model = est.causal_graph_
            except ImportError:
                # legacy pgmpy (< 1.0)
                from pgmpy.estimators import HillClimbSearch

                est = HillClimbSearch(data=data)
                model = est.estimate(
                    scoring_method=candidate,
                    max_indegree=max_parents,
                    max_iter=max_iter,
                    black_list=black_list,
                    show_progress=False,
                )
            return sorted(model.get_markov_blanket(node))
        except (TypeError, ValueError, KeyError) as e:  # unknown scoring name etc.
            last_err = e
            continue
    raise ValueError(
        f"none of the score aliases {_SCORE_ALIASES.get(score)} work with this pgmpy version"
    ) from last_err


@CORRELATORS.decorator("bn")
class BNCorrelation:
    """Per-node hill-climb BN learning -> Markov blankets as correlated sets."""

    def extract(
        self, frame: pd.DataFrame, config: BNCorrelationConfig | None = None
    ) -> CorrelationGraph:
        cfg = config or BNCorrelationConfig()
        try:
            import pgmpy  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "BNCorrelation requires the 'bn' extra: pip install 'nesydep[bn]'"
            ) from e

        nodes = [str(c) for c in frame.columns]
        data = frame.copy()
        data.columns = nodes
        targets = [c for c in nodes if data[c].nunique() > 1]

        sets: list[CorrelatedSet] = []
        for target in targets:
            # Forbid every edge that does not touch the target: the learned
            # graph then isolates the target's direct dependency structure.
            black_list = [
                (x, y) for x in nodes for y in nodes if x != y and x != target and y != target
            ]
            blanket = _learn_markov_blanket((data, target, cfg.score, black_list, int(1e4), cfg.max_parents))
            antecedents = tuple(a for a in blanket if a != target)
            if antecedents:
                sets.append(CorrelatedSet(antecedents=antecedents, target=target))
        return CorrelationGraph(sets)
