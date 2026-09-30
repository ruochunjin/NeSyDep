# NeSyDep

**Ne**ural-**Sy**mbolic **Dep**endency discovery — fast mining of functional
dependencies (FDs) and conditional functional dependencies (CFDs), built from
the research code behind two papers:

- *Fast Discovery of Functional Dependencies via Bayesian Network Learning* (ICDE 2026) → `algo="bsfd"`
- *Fast Discovery of CFDs via Transformer-Guided Relation Partitioning* → `algo="scfdm"`

## Why NeSyDep

- **One line to results**: `nesydep.discover(df, algo="bsfd")`.
- **Pluggable pipeline**: sampling → correlation extraction → partitioning →
  mining → evaluation. Swap any stage by name.
- **Fast native kernels**: TANE/DFD/CTane and the SCFDM strategies run in C++
  via pybind11; sub-table parallelism via Python process pools.
- **Works without a GPU**: the Transformer correlator falls back to
  lightweight statistical correlation.

See [Quickstart](quickstart.md) to get running in five minutes.
