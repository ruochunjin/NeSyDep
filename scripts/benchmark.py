"""Benchmark the algorithms on the frozen samples; writes docs/benchmarks.md.

Run:  python scripts/benchmark.py
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import nesydep as nd  # noqa: E402

DATA = ROOT / "tests" / "data"


def bench(name: str, algo: str, data: pd.DataFrame, **params) -> dict:
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        result = nd.discover(data, algo=algo, **params)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "dataset": name, "algo": algo, "n": len(result),
            "seconds": round(elapsed, 2), "peak_mb": round(peak / 1e6, 1),
            "note": "",
        }
    except Exception as e:
        tracemalloc.stop()
        return {
            "dataset": name, "algo": algo, "n": "-",
            "seconds": round(time.perf_counter() - t0, 2), "peak_mb": "-",
            "note": f"{type(e).__name__} (guarded)",
        }


def main() -> None:
    toy = pd.read_csv(DATA / "toy_addresses.csv")
    german = pd.read_csv(DATA / "german_credit_sample.csv")
    census = pd.read_csv(DATA / "census42_sample.csv")
    alarm = pd.read_csv(DATA / "alarm_sample.csv")

    rows = [
        bench("toy (12×6)", "tane", toy, support=1, confidence=1.0),
        bench("toy (12×6)", "dfd", toy, support=1, confidence=1.0),
        bench("german16 (300×16)", "tane", german.iloc[:, :16], support=5, confidence=1.0),
        bench("german (300×21)", "bsfd", german, support=5, confidence=0.95),
        bench("alarm (500×37)", "bsfd", alarm, support=10, confidence=0.95),
        bench("census42 (500×42)", "scfdm", census, support=10, confidence=0.9, max_lhs=2),
        bench("census42 (500×42)", "ctane", census, support=10, confidence=0.9, max_lhs=2),
        # guard demonstration: full-table TANE on 42 columns
        bench("census42 (500×42)", "tane", census, support=10, confidence=0.95),
    ]

    lines = [
        "# Benchmarks",
        "",
        "Machine-local numbers (development laptop, MinGW build). Regenerate with "
        "`python scripts/benchmark.py`. The last row shows the search-space guard "
        "refusing unbounded full-table mining on a wide table.",
        "",
        "| Dataset | Algorithm | Dependencies | Time (s) | Peak RAM (MB) | Note |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['algo']} | {r['n']} | {r['seconds']} | "
            f"{r['peak_mb']} | {r['note']} |"
        )
    out = ROOT / "docs" / "benchmarks.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
