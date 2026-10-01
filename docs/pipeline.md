# The five-stage pipeline

Every fast dependency-discovery method in NeSyDep follows the same shape:

```
Sampler ──► CorrelationExtractor ──► Partitioner ──► Miner ──► Evaluator
(rows)        (column groups)         (sub-tables)     (deps)     (scores)
```

| Stage | What it does | Built-ins |
|---|---|---|
| Sampler | row-level reduction | `random`, `stratified` (BSFD), `representative` (RepSampler, `[lsh]` extra) |
| CorrelationExtractor | column-level reduction | `lightweight` (Pearson/PCA/Lasso), `bn` (SubLearner, `[bn]` extra), `transformer` (AttrFinder, `[transformer]` extra) |
| Partitioner | vertical split into sub-tables | `vertical` |
| Miner | dependency discovery | `pfminer`, `tane`, `dfd`, `scfdm`, `ctane`, `fastafd` (native), `pyref-fd` (pure Python) |
| Evaluator | scoring vs ground truth | P/R/F1 (`nesydep.evaluate`) |

Algorithm presets (`bsfd`, `scfdm`) are just named combinations of stages.
You can compose your own:

```python
from nesydep import Pipeline

pipe = Pipeline(
    sampler="stratified",
    correlation="bn",
    partitioner="vertical",
    miner="pfminer",
    configs={"miner": cfg},
    cache_dir="run1/",   # intermediates land on disk for reproducibility
)
result = pipe.run(df)
```

End-to-end baselines (`tane`, `dfd`, `ctane`, `pyref-fd`) are pipelines with
identity sampling/correlation/partitioning stages.

## Auto modes

Paper mechanisms hide behind `"auto"` sentinels:

- `RepresentativeSampleConfig(bound="auto")` computes the minimum sample size
  from the data via the paper's Theorem-1/2 sampling lower bounds.
- `TransformerCorrelationConfig(threshold="auto")` uses the
  normalised-Shannon-entropy adaptive retention threshold.
