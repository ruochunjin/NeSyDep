# Changelog

## v0.1.2 (2026-10-01)

- Fixed wheel-test workflow: test module no longer shadows the installed
  wheel with the source tree; the wheels CI now asserts the native extension
  loads before running the golden regression.

## v0.1.1 (2026-09-30)

- C++17 portability fix (std::ptr_fun removed in libc++/MSVC) — wheels now
  build on all three platforms (v0.1.0's tag predated the fix).
- CI fully green: ruff + mypy (pinned, scoped to core/io) + 3 OS × 4 Python
  versions.

## v0.1.0 (2026-09-30)

First public milestone (MVP).

### Algorithms

- **BSFD** — fast FD discovery: stratified sampling → SubLearner
  (Bayesian-network Markov blankets) → vertical partitioning → native
  PFMiner kernel. Paper: *Fast Discovery of FDs via Bayesian Network
  Learning* (ICDE 2026).
- **SCFDM** — fast CFD discovery: RepSampler (MinHash-LSH, with the paper's
  Theorem-1/2 sampling lower bounds via `bound="auto"`) → correlation
  extraction (Transformer AttrFinder with entropy-adaptive thresholds, or
  lightweight Pearson/PCA/Lasso) → native CFD kernel with all 10 search
  strategies.
- **Baselines**: TANE, DFD (native), CTane (native, `Integrated-BFS`
  strategy), plus `pyref-fd`, a pure-Python textbook reference miner.

### Platform

- Five-stage pluggable pipeline (Sampler → CorrelationExtractor →
  Partitioner → Miner → Evaluator) with name-based registries and
  entry-point plugin support.
- Native C++17 kernels via pybind11 (`nesydep._core`); sub-table parallelism
  in Python process pools; GIL released during mining.
- Unified FD/CFD models, legacy text formats compatible with the original
  research code, JSON/CSV/parquet export.
- Evaluation: P/R/F1 metrics, CFD/FD support-confidence verification
  (CTane tuple-removal semantics for variable CFDs).
- CLI: `nesydep discover / sample / verify / evaluate / algorithms`.
- 53 tests including a golden regression suite with multi-source
  cross-validation (see `tests/regression/README_golden.md`).

### Notes

- The native TANE/DFD kernels use pair-based support/confidence
  (`sup/(n²−n)`, prototype semantics); key-column FDs are invisible to them.
- Prototype DFD is approximate (can miss valid minimal FDs); TANE is the
  reference implementation.
- Optional extras: `[bn]` (pgmpy), `[transformer]` (torch), `[lsh]`
  (datasketch), `[all]`.
