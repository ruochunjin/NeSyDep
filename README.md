# NeSyDep

**Ne**ural-**Sy**mbolic **Dep**endency discovery: fast mining of data dependencies
(functional dependencies, conditional functional dependencies, and more to come).

NeSyDep turns the research prototypes behind our papers into a usable, extensible
open-source tool for data scientists and data engineers:

- **BSFD** — fast FD discovery guided by Bayesian-network structure learning
  ([paper](https://github.com/Y-Staring/BSFD), ICDE 2026)
- **SCFDM** — fast CFD discovery via Transformer-guided relation partitioning
  ([paper](https://github.com/chen760316/CFDs-2026))

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

If you use NeSyDep in academic work, please cite the papers listed in
[CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
