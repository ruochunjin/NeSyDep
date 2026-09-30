"""Mining results: container + (de)serialisation.

``to_legacy_txt()`` reproduces the prototypes' text formats so refactored
components can be diffed against frozen golden outputs from the original
research code (see ``tests/regression/README_golden.md``).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    import pandas as pd

from nesydep.core.dependency import CFD, FD, Dependency


@dataclass
class MiningResult:
    """The output of one discovery run.

    Attributes:
        algorithm: registered name of the algorithm that produced the result.
        dependencies: discovered dependencies (FDs, CFDs, ...).
        stats: free-form measurements (stage timings, sub-table counts, ...).
        config_snapshot: full resolved config dict, for reproducibility.
    """

    algorithm: str
    dependencies: list[Dependency] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    # -- views ---------------------------------------------------------------

    @property
    def fds(self) -> list[FD]:
        return [d for d in self.dependencies if isinstance(d, FD)]

    @property
    def cfds(self) -> list[CFD]:
        return [d for d in self.dependencies if isinstance(d, CFD)]

    def __len__(self) -> int:
        return len(self.dependencies)

    def __iter__(self) -> Iterable[Dependency]:
        return iter(self.dependencies)

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "created_at": self.created_at,
            "stats": self.stats,
            "config_snapshot": self.config_snapshot,
            "dependencies": [d.to_dict() for d in self.dependencies],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MiningResult:
        return cls(
            algorithm=data["algorithm"],
            dependencies=[Dependency.from_dict(d) for d in data.get("dependencies", [])],
            stats=data.get("stats", {}),
            config_snapshot=data.get("config_snapshot", {}),
            created_at=data.get("created_at", time.time()),
        )

    def to_json(self, path: str | Path, **json_kwargs: Any) -> None:
        kwargs: dict[str, Any] = {"indent": 2, "ensure_ascii": False}
        kwargs.update(json_kwargs)
        Path(path).write_text(json.dumps(self.to_dict(), **kwargs), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> MiningResult:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_pandas(self) -> pd.DataFrame:
        """One row per dependency; CFD pattern columns are NaN for plain FDs."""
        import pandas as pd

        return pd.DataFrame([d.to_dict() for d in self.dependencies])

    def to_csv(self, path: str | Path) -> None:
        self.to_pandas().to_csv(path, index=False)

    def to_parquet(self, path: str | Path) -> None:
        self.to_pandas().to_parquet(path, index=False)

    def to_legacy_txt(self, path: str | Path) -> None:
        """Write in the prototypes' text format (one dependency per line).

        FD:    ``[A, B] -> C``
        CFD:   ``[A, B] => C, (a1, b1 || c1)``
        """
        lines = [str(d) for d in self.dependencies]
        Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
