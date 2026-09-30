# Golden regression references

Golden outputs are the correctness safety net for the mining kernels. Every
golden was frozen by `freeze_golden.py` **only after independent
cross-checks passed** — the checks are the point, the files are just the
record.

## Provenance and validation per artifact

| Golden | Input | Reference | Cross-validation applied |
|---|---|---|---|
| `fd_pyref_toy.txt` | toy_addresses.csv (12×6) | pure-Python textbook miner | hand-verified ground truth in `tests/data/toy_ground_truth.py` ⊆ output; every FD exact |
| `fd_tane_toy.txt` | toy_addresses.csv | native TANE | DFD agrees exactly |
| `fd_tane_german16.txt` | german_credit_sample.csv first 16 cols | native TANE | every FD brute-force verified to **hold exactly and be minimal**; DFD ⊆ TANE |
| `fd_pipeline_german.txt` / `fd_pipeline_census42.txt` | german_credit / census42 samples | seeded pipeline (stratified + Pearson + vertical + PFMiner) | deterministic reproduction; FDs mined on the sample are the reference (approximate on the full table by design) |
| `cfd_fd-first_*.txt` | toy / german / census42 | SCFDM FD-First family | all 4 FD-First strategies agree; every CFD re-verified on full sample with independent pandas checker |
| `cfd_ctane_*.txt` | toy / german / census42 | CTane family (Itemset-First ×4, Integrated-DFS, Integrated-BFS) | all 6 strategies agree (after excluding trivial empty-LHS CFDs); same verification |

## Semantics notes (learned while validating)

- The prototype TANE/DFD use pair-based support/confidence (`sup/(n²−n)`);
  FDs whose LHS groups are all singletons (key columns) are invisible to
  them. The textbook miner (`pyref-fd`) does find those.
- The prototype DFD is approximate: on german16 it misses 63 valid minimal
  FDs that TANE finds (all brute-force confirmed). TANE is the FD reference.
- CFD strategies form two families with different minimality notions:
  FD-First (finer, more constant CFDs) vs CTane-style Itemset-First /
  Integrated (coarser cover). Agreement is only expected within a family.
- Integrated-DFS occasionally emits trivial empty-LHS CFDs `() => A, (_)`;
  excluded from comparison.
- Full-table exact mining is exponential in columns for this simplified
  TANE kernel (german_credit 21 cols overflows memory); that is the problem
  BSFD's search-space reduction solves — wide-table goldens therefore use
  the partitioned pipeline.

## Regenerating

```bash
python tests/regression/freeze_golden.py   # writes golden/ + manifest.json
python -m pytest tests/regression          # verifies against frozen goldens
```

Only commit regenerated goldens after reviewing the diff — a changed golden
means changed mining behavior.

To additionally cross-validate against the *original prototype* outputs
(e.g. a paper reproduction run on Linux with MPI), drop the prototype output
files here as `prototype_*.txt` and compare with `nesydep.io.legacy` parsing.
