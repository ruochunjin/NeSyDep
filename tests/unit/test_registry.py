"""Registry behaviour, including entry-point plugin discovery."""

import pytest

from nesydep.core.registry import ALGORITHMS, MINERS, Registry


def test_register_and_get():
    reg: Registry[type] = Registry("thing")
    reg.register("x", int)
    assert reg.get("x") is int
    assert "x" in reg.names()


def test_duplicate_registration_rejected():
    reg: Registry[type] = Registry("thing")
    reg.register("x", int)
    with pytest.raises(KeyError):
        reg.register("x", float)
    reg.register("x", float, replace=True)
    assert reg.get("x") is float


def test_unknown_name_lists_available():
    reg: Registry[type] = Registry("thing")
    reg.register("x", int)
    with pytest.raises(KeyError, match="'y'.*x"):
        reg.get("y")


def test_builtin_algorithms_registered():
    names = ALGORITHMS.names()
    for expected in ("bsfd", "scfdm", "tane", "dfd", "ctane", "pyref-fd"):
        assert expected in names


def test_builtin_miners_registered():
    names = MINERS.names()
    for expected in ("pfminer", "tane", "dfd", "scfdm", "ctane", "pyref-fd"):
        assert expected in names
