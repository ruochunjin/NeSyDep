# Quickstart

## Install

```bash
pip install nesydep                  # core
pip install "nesydep[bn]"            # Bayesian-network correlation (BSFD default)
pip install "nesydep[transformer]"   # AttrFinder (large: pulls torch)
pip install "nesydep[lsh]"           # RepSampler (MinHash-LSH)
```

## Discover FDs

```python
import pandas as pd
import nesydep as nd

df = pd.read_csv("hospital.csv")
result = nd.discover(df, algo="bsfd", support=100, confidence=0.95)
for fd in result.fds:
    print(fd)
result.to_json("fds.json")
```

## Discover CFDs

```python
miner = nd.get_algorithm("scfdm")(correlation="lightweight", support=5, confidence=0.9)
result = miner.discover(df)
result.to_legacy_txt("cfds.txt")   # [A, B] => C, (a, b || c)
```

## Verify against the full dataset

```python
from nesydep.core.dataset import as_categorical_view, as_dataset
from nesydep.evaluation.verify import verify

frame = as_categorical_view(as_dataset("hospital_full.csv"))
checks = verify(frame, result.dependencies, min_support=100, min_confidence=0.95)
print(f"{sum(v.holds for v in checks)}/{len(checks)} hold")
```

## CLI

```bash
nesydep discover data.csv --algo bsfd --support 100 -o fds.csv
nesydep sample data.csv --method stratified --ratio 0.15 -o sample.csv
nesydep verify cfds.txt --data full.csv --min-support 100
nesydep evaluate result.txt --ground-truth gold.txt
nesydep algorithms
```
