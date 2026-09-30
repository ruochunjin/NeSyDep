"""Text (de)serialisation for dependency formats.

The CFD line format is byte-compatible with the SCFDM C++ miners'
``Output::printCFD`` and the prototype ``verify.py`` parser:

    [A, B] => C, (a1, b1 || c1)

The FD line format follows the same convention without patterns:

    [A, B] -> C
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from nesydep.core.dependency import CFD, FD, Dependency

# Identical to the prototype's verify.py / cfd_parser.py pattern.
_CFD_PATTERN = re.compile(r"\[(.*?)\]\s*=>\s*(.*?),\s*\((.*?)\|\|(.*?)\)")
_FD_PATTERN = re.compile(r"\[(.*?)\]\s*->\s*(.+?)\s*$")


def _split_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip() != ""]


def parse_fd(line: str) -> FD | None:
    """Parse one ``[A, B] -> C`` line; return None if it doesn't match."""
    m = _FD_PATTERN.search(line.strip())
    if not m:
        return None
    return FD(lhs=tuple(_split_list(m.group(1))), rhs=m.group(2).strip())


def parse_cfd(line: str) -> CFD | None:
    """Parse one ``[A, B] => C, (a1, b1 || c1)`` line; None if no match."""
    m = _CFD_PATTERN.search(line.strip())
    if not m:
        return None
    return CFD(
        lhs=tuple(_split_list(m.group(1))),
        rhs=m.group(2).strip(),
        lhs_pattern=tuple(_split_list(m.group(3))),
        rhs_pattern=m.group(4).strip(),
    )


def parse_line(line: str) -> Dependency | None:
    """Parse a line as CFD first (it is the more specific format), then FD."""
    return parse_cfd(line) or parse_fd(line)


def load_dependencies(path: str | Path) -> list[Dependency]:
    """Load a legacy text file, skipping blank/unparseable lines."""
    deps: list[Dependency] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        dep = parse_line(line)
        if dep is not None:
            deps.append(dep)
    return deps


def dump_dependencies(deps: Iterable[Dependency], path: str | Path) -> None:
    lines = [str(d) for d in deps]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def normalize_key(dep: Dependency) -> tuple:
    """Order-invariant hashable key for set comparisons (regression tests).

    The models already sort/dedup LHS on construction, so the key is just a
    canonical tuple; it exists as a named hook for future kinds whose
    normalisation is non-trivial.
    """
    if isinstance(dep, CFD):
        return ("cfd", tuple(zip(dep.lhs, dep.lhs_pattern)), dep.rhs, dep.rhs_pattern)
    if isinstance(dep, FD):
        return ("fd", dep.lhs, dep.rhs)
    return (dep.kind, repr(sorted(dep.to_dict().items())))
