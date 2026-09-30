"""Unified data-dependency models.

All dependencies are immutable dataclasses so they can live in sets and serve
as dict keys. ``FD`` covers plain functional dependencies; ``CFD`` adds a
pattern tuple (``"_"`` is the wildcard / variable marker, matching the
prototypes' text format ``[A, B] => C, (a1, b1 || c1)``).

New dependency kinds (OD, DC, UC, GCFD, ...) subclass :class:`Dependency`
and register their ``kind`` — nothing else in the core needs to change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

WILDCARD = "_"


class Dependency:
    """Base class for all dependency types."""

    kind: ClassVar[str] = "dependency"

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dependency:
        kind = data.get("kind")
        for sub in _KIND_REGISTRY.values():
            if sub.kind == kind:
                return sub._from_dict(data)
        raise ValueError(f"unknown dependency kind: {kind!r}")

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Dependency:
        raise NotImplementedError


_KIND_REGISTRY: dict[str, type[Dependency]] = {}


def register_kind(cls: type[Dependency]) -> type[Dependency]:
    """Class decorator: make a Dependency subclass (de)serializable by kind."""
    _KIND_REGISTRY[cls.kind] = cls
    return cls


@register_kind
@dataclass(frozen=True)
class FD(Dependency):
    """A functional dependency ``lhs -> rhs``.

    ``lhs`` is stored sorted and deduplicated so equality is order-invariant.
    """

    lhs: tuple[str, ...]
    rhs: str

    kind: ClassVar[str] = "fd"

    def __post_init__(self) -> None:
        object.__setattr__(self, "lhs", tuple(sorted(set(self.lhs))))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "lhs": list(self.lhs), "rhs": self.rhs}

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> FD:
        return cls(lhs=tuple(data["lhs"]), rhs=data["rhs"])

    def __str__(self) -> str:
        return f"[{', '.join(self.lhs)}] -> {self.rhs}"


@register_kind
@dataclass(frozen=True)
class CFD(Dependency):
    """A conditional functional dependency ``lhs => rhs, (lhs_pattern || rhs_pattern)``.

    Pattern values are strings; :data:`WILDCARD` marks a variable position.
    ``lhs_pattern[i]`` is the pattern for ``lhs[i]`` — the two tuples stay
    aligned, so LHS normalisation sorts them together.
    """

    lhs: tuple[str, ...]
    rhs: str
    lhs_pattern: tuple[str, ...]
    rhs_pattern: str

    kind: ClassVar[str] = "cfd"

    def __post_init__(self) -> None:
        if len(self.lhs) != len(self.lhs_pattern):
            raise ValueError(
                f"lhs ({len(self.lhs)}) and lhs_pattern ({len(self.lhs_pattern)}) "
                "must have the same length"
            )
        # Sort LHS attributes together with their patterns so that
        # [A,B]=>(a,b) and [B,A]=>(b,a) normalise to the same object.
        pairs = sorted(zip(self.lhs, self.lhs_pattern), key=lambda p: p[0])
        deduped: dict[str, str] = {}
        for attr, pat in pairs:
            if attr in deduped and deduped[attr] != pat:
                raise ValueError(f"conflicting patterns for attribute {attr!r}")
            deduped[attr] = pat
        object.__setattr__(self, "lhs", tuple(deduped))
        object.__setattr__(self, "lhs_pattern", tuple(deduped[a] for a in self.lhs))

    @property
    def is_constant(self) -> bool:
        """True when no position uses the wildcard (a constant CFD)."""
        return self.rhs_pattern != WILDCARD and all(p != WILDCARD for p in self.lhs_pattern)

    def to_fd(self) -> FD:
        """The embedded FD (patterns dropped)."""
        return FD(lhs=self.lhs, rhs=self.rhs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "lhs": list(self.lhs),
            "rhs": self.rhs,
            "lhs_pattern": list(self.lhs_pattern),
            "rhs_pattern": self.rhs_pattern,
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> CFD:
        return cls(
            lhs=tuple(data["lhs"]),
            rhs=data["rhs"],
            lhs_pattern=tuple(data["lhs_pattern"]),
            rhs_pattern=data["rhs_pattern"],
        )

    def __str__(self) -> str:
        pats = ", ".join(self.lhs_pattern)
        return f"[{', '.join(self.lhs)}] => {self.rhs}, ({pats} || {self.rhs_pattern})"
