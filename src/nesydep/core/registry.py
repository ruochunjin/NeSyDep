"""Registries: the plugin backbone.

Every stage implementation and every algorithm registers under a string name.
Third-party packages can contribute algorithms via the ``nesydep.miners`` /
``nesydep.samplers`` / ... entry-point groups without touching this package.
"""
from __future__ import annotations

from importlib import metadata
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A name -> implementation mapping with plugin discovery."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}
        self._plugins_loaded = False

    def register(self, name: str, item: T, *, replace: bool = False) -> T:
        if name in self._items and not replace:
            raise KeyError(
                f"{self.kind} {name!r} is already registered; pass replace=True to override"
            )
        self._items[name] = item
        return item

    def decorator(self, name: str) -> Callable[[T], T]:
        def wrap(item: T) -> T:
            return self.register(name, item)

        return wrap

    def get(self, name: str) -> T:
        self._load_plugins()
        try:
            return self._items[name]
        except KeyError:
            available = ", ".join(sorted(self._items)) or "(none)"
            raise KeyError(
                f"unknown {self.kind} {name!r}. Available: {available}"
            ) from None

    def names(self) -> list[str]:
        self._load_plugins()
        return sorted(self._items)

    def items(self) -> dict[str, T]:
        self._load_plugins()
        return dict(self._items)

    def _load_plugins(self) -> None:
        """Pull in third-party entry points, once."""
        if self._plugins_loaded:
            return
        self._plugins_loaded = True
        group = f"nesydep.{self.kind}s"
        for ep in metadata.entry_points(group=group):
            if ep.name not in self._items:
                self._items[ep.name] = ep.load()


SAMPLERS: Registry[Any] = Registry("sampler")
CORRELATORS: Registry[Any] = Registry("correlation")
PARTITIONERS: Registry[Any] = Registry("partitioner")
MINERS: Registry[Any] = Registry("miner")
ALGORITHMS: Registry[Any] = Registry("algorithm")
