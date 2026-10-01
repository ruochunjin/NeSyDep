# NeSyDep

**Ne**ural-**Sy**mbolic **Dep**endency discovery: fast mining of data dependencies
(functional dependencies, conditional functional dependencies, and more to come).

NeSyDep turns the research prototypes behind our papers into a usable, extensible
open-source tool for data scientists and data engineers:

- **BSFD** — fast FD discovery guided by Bayesian-network structure learning (ICDE 2026)
- **SCFDM** — fast CFD discovery via Transformer-guided relation partitioning (VLDB 2027)

Mining kernels are C++ (exposed through pybind11); orchestration, sampling,
correlation extraction and evaluation are pure Python.

> Status: early development. See the roadmap below.

## Install

```bash
pip install nesydep            # core (FD/CFD mining + lightweight correlation)
pip install "nesydep[bn]"      # + Bayesian-network correlation (BSFD default)
pip install "nesydep[transformer]"  # + Transformer AttrFinder (large, pulls torch)
pip install "nesydep[all]"     # everything
```

## Quick start

```python
import pandas as pd
import nesydep as nd

df = pd.read_csv("hospital.csv")

# One-liner
fds = nd.discover(df, algo="bsfd", support=100, confidence=0.95)

# Full control
miner = nd.SCFDM(correlation="pearson", strategy="FD-First-DFS-dfs",
                 support=0.001, confidence=0.9)
result = miner.discover(df)
result.to_csv("cfds.csv")
```

CLI:

```bash
nesydep discover data.csv --algo bsfd -o fds.csv
nesydep algorithms            # list registered algorithms and parameters
```

## Roadmap

- **v0.1 (MVP)**: BSFD, SCFDM (lightweight correlation), TANE/DFD/CTane baselines,
  evaluation, CLI.
- **v0.2**: Transformer AttrFinder, all 10 SCFDM strategies, prebuilt wheels.
- **v1.0**: PyPI release, full docs, plugin tutorial.
- Later: GCFD and other dependency types (OD, DC, UC) as plugins.

Out of scope for v1: direct database connections, distributed (MPI) wheels.

## Citing

If you use NeSyDep in academic work, please cite:

**CFD discovery (SCFDM):**

> Ruochun Jin, Shenglin Chen, Xi Wang, Siyi Yang, Ting Wang. Fast Discovery of Conditional Functional Dependencies via Transformer-Guided Relation Partitioning. VLDB 2027.

**FD discovery (BSFD):**

```bibtex
@inproceedings{yang2026fast,
  title={Fast Discovery of Functional Dependencies via Bayesian Network Learning},
  author={Yang, Siyi and Chen, Shenglin and Wang, Xi and Tang, Yuhua and Jin, Ruochun},
  booktitle={2026 IEEE 42nd International Conference on Data Engineering (ICDE)},
  pages={44--57},
  year={2026},
  organization={IEEE}
}
```

**Related work:**

```bibtex
@inproceedings{wang2024boosting,
  title={Boosting meaningful dependency mining with clustering and covariance analysis},
  author={Wang, Xi and Jin, Ruochun and Huang, Wanrong and Tang, Yuhua},
  booktitle={2024 IEEE 40th International Conference on Data Engineering (ICDE)},
  pages={639--652},
  year={2024},
  organization={IEEE}
}
```

## License

MIT — see [LICENSE](LICENSE).
