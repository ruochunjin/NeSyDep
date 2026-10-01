# Writing a plugin

NeSyDep's algorithms and stages live in registries. Third-party packages add
implementations through entry points — no fork needed.

## A new miner

```python
# my_package/miners.py
from nesydep.core.stages import SubTable

class MyMiner:
    def mine(self, subtables: list[SubTable], config) -> list:
        ...  # return nesydep Dependency objects (FD, CFD, ...)
```

```toml
# my_package/pyproject.toml
[project.entry-points."nesydep.miners"]
myminer = "my_package.miners:MyMiner"
```

After `pip install my_package`, `Pipeline(miner="myminer")` just works.

Entry-point groups: `nesydep.samplers`, `nesydep.correlations`,
`nesydep.partitioners`, `nesydep.miners`, `nesydep.algorithms`.

## A new algorithm preset

Subclass `nesydep.algorithms.MiningAlgorithm`: define `_default_config()` and
`_pipeline()`, then register (decorator or entry point):

```python
from nesydep.algorithms import MiningAlgorithm
from nesydep.core.registry import ALGORITHMS

@ALGORITHMS.decorator("myalgo")
class MyAlgo(MiningAlgorithm):
    name = "myalgo"
    def _default_config(self): ...
    def _pipeline(self): ...
```

## A new dependency kind (e.g. OD, DC, UC)

1. Subclass `nesydep.core.dependency.Dependency`, set a unique `kind`,
   implement `to_dict`/`_from_dict`, and decorate with `@register_kind`.
2. Implement the stages you need (often just a `Miner`).
3. Combine into a `MiningAlgorithm` preset.

The FDX integration (forthcoming) will be the worked example for this path.
