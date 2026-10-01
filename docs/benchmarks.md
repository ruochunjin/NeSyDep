# Benchmarks

Machine-local numbers (development laptop, MinGW build). Regenerate with `python scripts/benchmark.py`. The last row shows the search-space guard refusing unbounded full-table mining on a wide table.

| Dataset | Algorithm | Dependencies | Time (s) | Peak RAM (MB) | Note |
|---|---|---|---|---|---|
| toy (12×6) | tane | 6 | 0.09 | 0.3 |  |
| toy (12×6) | dfd | 6 | 0.0 | 0.0 |  |
| german16 (300×16) | tane | 113 | 1.53 | 0.2 |  |
| german (300×21) | bsfd | 6 | 19.91 | 128.2 |  |
| alarm (500×37) | bsfd | 2608 | 48.82 | 7.5 |  |
| census42 (500×42) | scfdm | 1337 | 0.25 | 2.0 |  |
| census42 (500×42) | ctane | 538 | 0.11 | 0.8 |  |
| census42 (500×42) | tane | - | 0.03 | - | SearchSpaceExplosionError (guarded) |
