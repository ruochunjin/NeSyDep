"""``nesydep`` command-line interface (typer)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="nesydep",
    help="NeSyDep: fast neural-symbolic discovery of data dependencies (FD/CFD).",
    no_args_is_help=True,
)


@app.command()
def discover(
    data: Path = typer.Argument(..., help="Input table (CSV or parquet)."),
    algo: str = typer.Option("bsfd", "--algo", "-a", help="Registered algorithm name."),
    support: Optional[int] = typer.Option(None, help="Minimum support (rows)."),
    confidence: Optional[float] = typer.Option(None, help="Minimum confidence in (0, 1]."),
    max_lhs: Optional[int] = typer.Option(None, help="Maximum LHS size (0 = unlimited)."),
    strategy: Optional[str] = typer.Option(None, help="CFD search strategy (scfdm/ctane)."),
    correlation: Optional[str] = typer.Option(
        None, help="Correlation extractor for scfdm: transformer|lightweight."
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file."),
    fmt: str = typer.Option("txt", "--format", "-f", help="txt | csv | json | parquet"),
) -> None:
    """Discover dependencies in DATA and write them out."""
    import nesydep as nd

    params = {
        k: v
        for k, v in {
            "support": support,
            "confidence": confidence,
            "max_lhs": max_lhs,
            "strategy": strategy,
        }.items()
        if v is not None
    }
    cls = nd.get_algorithm(algo)
    if correlation is not None:
        from nesydep.core.config import (
            LightweightCorrelationConfig,
            TransformerCorrelationConfig,
        )

        corr_cfg = (
            TransformerCorrelationConfig()
            if correlation == "transformer"
            else LightweightCorrelationConfig(method=correlation)
        )
        algo_obj = cls(**params)
        algo_obj.config.correlation = corr_cfg
    else:
        algo_obj = cls(**params)

    result = algo_obj.discover(str(data))
    _write_result(result, output, fmt)
    typer.echo(f"discovered {len(result)} dependencies "
               f"(mine: {result.stats.get('mine_seconds', 0):.2f}s)")


@app.command()
def algorithms() -> None:
    """List registered algorithms and their key parameters."""
    import nesydep as nd

    for name in nd.list_algorithms():
        typer.echo(f"- {name}")


@app.command()
def evaluate(
    result: Path = typer.Argument(..., help="Mined dependencies (legacy txt)."),
    ground_truth: Path = typer.Option(..., "--ground-truth", "-g"),
) -> None:
    """Precision/recall/F1 of a result file against ground truth."""
    import nesydep as nd
    from nesydep.io.legacy import load_dependencies

    metrics = nd.evaluate(load_dependencies(result), load_dependencies(ground_truth))
    typer.echo(
        f"precision={metrics.precision:.4f} recall={metrics.recall:.4f} "
        f"f1={metrics.f1:.4f} (mined={metrics.n_mined}, truth={metrics.n_ground_truth})"
    )


@app.command()
def verify(
    deps: Path = typer.Argument(..., help="Dependencies to verify (legacy txt)."),
    data: Path = typer.Option(..., "--data", "-d", help="Full dataset (CSV)."),
    min_support: int = typer.Option(1),
    min_confidence: float = typer.Option(1.0),
) -> None:
    """Check each dependency's support/confidence against the full dataset."""
    from nesydep.core.dataset import as_categorical_view, as_dataset
    from nesydep.evaluation.verify import verify as _verify
    from nesydep.io.legacy import load_dependencies

    frame = as_categorical_view(as_dataset(data))
    results = _verify(frame, load_dependencies(deps), min_support, min_confidence)
    held = sum(v.holds for v in results)
    typer.echo(f"{held}/{len(results)} dependencies hold "
               f"(support>={min_support}, confidence>={min_confidence})")
    for v in results:
        if not v.holds:
            typer.echo(f"  FAIL {v.dependency} (support={v.support}, conf={v.confidence:.3f})")


@app.command()
def sample(
    data: Path = typer.Argument(..., help="Input CSV."),
    method: str = typer.Option("representative", "--method", "-m"),
    ratio: float = typer.Option(0.15, help="Sampling ratio (random/stratified)."),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Sample a dataset with one of the registered samplers."""
    import pandas as pd

    from nesydep.core.config import RandomSampleConfig, StratifiedSampleConfig
    from nesydep.core.registry import SAMPLERS

    frame = pd.read_csv(data)
    cfg = (
        RandomSampleConfig(ratio=ratio)
        if method == "random"
        else StratifiedSampleConfig(ratio=ratio)
        if method == "stratified"
        else None
    )
    sampler = SAMPLERS.get(method)()
    out = sampler.sample(frame, cfg)
    out.to_csv(output, index=False)
    typer.echo(f"sampled {len(frame)} -> {len(out)} rows -> {output}")


def _write_result(result, output: Optional[Path], fmt: str) -> None:
    if output is None:
        for dep in result:
            typer.echo(str(dep))
        return
    writers = {
        "txt": result.to_legacy_txt,
        "csv": result.to_csv,
        "json": result.to_json,
        "parquet": result.to_parquet,
    }
    if fmt not in writers:
        raise typer.BadParameter(f"unknown format {fmt!r}; choose from {sorted(writers)}")
    writers[fmt](output)
