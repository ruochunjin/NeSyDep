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

# A small hospital table ships with the repo — load it straight from GitHub.
url = "https://raw.githubusercontent.com/ruochunjin/NeSyDep/main/examples/data/hospital_sample.csv"
df = pd.read_csv(url)

# One-liner FD discovery
fds = nd.discover(df, algo="bsfd", support=2, confidence=0.95)
for fd in fds.fds:
    print(fd)   # e.g. [Zip] -> City, [Zip] -> State, [Address] -> PhoneNumber

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

**Also from our group** (its algorithms will be integrated into NeSyDep in a future release):

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
