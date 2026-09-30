"""Dataset abstraction.

A :class:`Dataset` is a thin, typed wrapper around tabular data. CFD mining
semantics require string-typed values (pattern matching), so
:meth:`as_categorical_view` is the canonical entry into the mining engines.

Direct database connections are intentionally out of scope for v1 — read with
SQLAlchemy/pandas first, then wrap the DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DatasetLike(Protocol):
    """Anything the pipeline can treat as input data."""

    def to_pandas(self) -> pd.DataFrame: ...
    @property
    def columns(self) -> list[str]: ...
    @property
    def n_rows(self) -> int: ...


@dataclass
class PandasDataset:
    """In-memory dataset backed by a DataFrame."""

    frame: pd.DataFrame
    name: str = "dataframe"

    def to_pandas(self) -> pd.DataFrame:
        return self.frame

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self.frame.columns]

    @property
    def n_rows(self) -> int:
        return len(self.frame)


@dataclass
class CSVDataset:
    """Lazily loaded CSV file."""

    path: Path
    read_kwargs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def to_pandas(self) -> pd.DataFrame:
        return pd.read_csv(self.path, **self.read_kwargs)

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in pd.read_csv(self.path, nrows=0, **self.read_kwargs).columns]

    @property
    def n_rows(self) -> int:
        with open(self.path, "rb") as f:
            # header counts as one line
            return sum(1 for _ in f) - 1

    @property
    def name(self) -> str:
        return self.path.stem


@dataclass
class ParquetDataset:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def to_pandas(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)

    @property
    def columns(self) -> list[str]:
        import pyarrow.parquet as pq

        return [str(c) for c in pq.read_schema(self.path).names]

    @property
    def n_rows(self) -> int:
        import pyarrow.parquet as pq

        return int(pq.read_metadata(self.path).num_rows)


def as_dataset(data: DatasetLike | pd.DataFrame | str | Path) -> DatasetLike:
    """Coerce user input (DataFrame, path, Dataset) into a DatasetLike."""
    # Duck-typing instead of isinstance on the Protocol: protocols with
    # non-method members (properties) raise TypeError on isinstance() on
    # some supported Python versions.
    if all(hasattr(data, attr) for attr in ("to_pandas", "columns", "n_rows")):
        return data  # type: ignore[return-value]
    if isinstance(data, pd.DataFrame):
        return PandasDataset(data)
    path = Path(data)  # type: ignore[arg-type]  # narrowed by the duck-type guard above
    if path.suffix.lower() in {".parquet", ".pq"}:
        return ParquetDataset(path)
    return CSVDataset(path)


def as_categorical_view(dataset: DatasetLike) -> pd.DataFrame:
    """Materialise the dataset as an all-string DataFrame.

    Mining engines compare values for equality; normalising to ``str`` up
    front keeps ``1`` and ``"1"`` from silently diverging and matches the
    prototypes' C++ readers (which read everything as strings). Missing
    values become the empty string.
    """
    frame = dataset.to_pandas()
    frame = frame.astype("string").fillna("")
    frame.columns = [str(c) for c in frame.columns]
    return frame
