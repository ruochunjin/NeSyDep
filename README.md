# NeSyDep

[![PyPI](https://img.shields.io/pypi/v/nesydep)](https://pypi.org/project/nesydep/)
[![CI](https://github.com/ruochunjin/NeSyDep/actions/workflows/ci.yml/badge.svg)](https://github.com/ruochunjin/NeSyDep/actions/workflows/ci.yml)
[![wheels](https://github.com/ruochunjin/NeSyDep/actions/workflows/wheels.yml/badge.svg)](https://github.com/ruochunjin/NeSyDep/actions/workflows/wheels.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Ne**ural-**Sy**mbolic **Dep**endency discovery: fast mining of data dependencies
(functional dependencies, conditional functional dependencies, and more to come).

NeSyDep turns the research prototypes behind our papers into a usable, extensible
open-source tool for data scientists and data engineers:

- **BSFD** — fast FD discovery guided by Bayesian-network structure learning (ICDE 2026)
- **SCFDM** — fast CFD discovery via Transformer-guided relation partitioning (VLDB 2027)
- **FastAFD** — approximate FD mining (FDm) boosted by clustering and covariance
  analysis (ICDE 2024)

> **Which algorithm should I use?** Start with **BSFD** (FDs) or **SCFDM** (CFDs):
> in our benchmarks they deliver the highest speedups and the most complete rule
> sets. FastAFD is offered as a complementary approximate method.

Mining kernels are C++ (exposed through pybind11); orchestration, sampling,
correlation extraction and evaluation are pure Python.

## Features

- **One line to results**: `nesydep.discover(df, algo="bsfd")` — from a pandas
  DataFrame, a CSV file, or a parquet file.
- **Algorithms built in**: BSFD, SCFDM, FastAFD, plus baselines TANE, DFD,
  CTane (all native C++), and a pure-Python textbook reference miner
  (`pyref-fd`).
- **Pluggable pipeline**: sampling → correlation extraction → partitioning →
  mining → evaluation. Swap any stage by name, or register your own via
  entry points (see [docs/plugins.md](docs/plugins.md)).
- **Robust by design**: memory-safety guards stop combinatorial explosion on
  wide tables with actionable guidance; optional heavy dependencies
  (torch/pgmpy) degrade gracefully instead of failing.
- **Evaluation built in**: precision/recall/F1, exact and approximate
  (pair-based, FDm semantics) support/confidence verification.
- **Prebuilt wheels** for Linux, macOS (arm64) and Windows (x64),
  Python 3.10–3.13. Windows wheels need no VC++ redistributable.

## Install

```bash
pip install nesydep            # core (FD/CFD mining + lightweight correlation)
pip install "nesydep[bn]"      # + Bayesian-network correlation (BSFD default)
pip install "nesydep[lsh]"     # + RepSampler (MinHash-LSH, SCFDM default)
pip install "nesydep[transformer]"  # + Transformer AttrFinder (large, pulls torch)
pip install "nesydep[all]"     # everything
```

## Quick start

```python
import pandas as pd
import nesydep as nd

# A small hospital table ships with the repo — load it straight from GitHub.
url = "https://raw.githubusercontent.com/ruochunjin/NeSyDep/main/examples/data/hospital_sample.csv"
df = pd.read_csv(url)

# One-liner FD discovery
fds = nd.discover(df, algo="bsfd", support=2, confidence=0.95)
for fd in fds.fds:
    print(fd)   # e.g. [Zip] -> City, [Zip] -> State, [County] -> State

# CFD discovery with full control
miner = nd.SCFDM(correlation="pearson", strategy="FD-First-DFS-dfs",
                 support=2, confidence=0.9, max_lhs=2)
result = miner.discover(df)
result.to_csv("cfds.csv")
```

CLI:

```bash
curl -O https://raw.githubusercontent.com/ruochunjin/NeSyDep/main/examples/data/hospital_sample.csv
nesydep discover hospital_sample.csv --algo bsfd --support 2 -o fds.txt
nesydep algorithms            # list registered algorithms and parameters
```

More examples: [examples/](examples/) — executed notebooks for FD quickstart,
CFD with SCFDM, and custom pipelines/plugins.

## Roadmap

- **v0.1** ✅ BSFD, SCFDM (lightweight correlation), TANE/DFD/CTane baselines,
  evaluation, CLI.
- **v0.2** ✅ Transformer AttrFinder, all 10 SCFDM strategies, prebuilt wheels.
- **v1.0** ✅ PyPI release, memory-safety guards, benchmarks.
- **v1.1** ✅ FastAFD (FDm, ICDE 2024) integrated; static-CRT Windows wheels.
- **Later**: FDX (forthcoming from our group), more fast mining algorithms over
  relational data, and more dependency types (OD, DC, UC, ...) as plugins.

Out of scope: GCFD (graph rules), direct database connections,
distributed (MPI) wheels.

## Citing

If you use NeSyDep in academic work, please cite:

**CFD discovery (SCFDM):**

```bibtex
@article{jin2027fast,
  title={Fast Discovery of Conditional Functional Dependencies via Transformer-Guided Relation Partitioning},
  author={Jin, Ruochun and Chen, Shenglin and Wang, Xi and Yang, Siyi and Wang, Ting},
  journal={Proceedings of the VLDB Endowment},
  year={2027}
}
```

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

**FastAFD (also from our group; integrated since v1.1.0):**

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
