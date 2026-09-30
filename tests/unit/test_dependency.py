"""Unit tests for the dependency models."""

import pytest

from nesydep.core.dependency import CFD, FD, WILDCARD, Dependency


class TestFD:
    def test_lhs_sorted_and_deduped(self):
        fd = FD(lhs=("b", "a", "b"), rhs="c")
        assert fd.lhs == ("a", "b")

    def test_order_invariant_equality(self):
        assert FD(lhs=("a", "b"), rhs="c") == FD(lhs=("b", "a"), rhs="c")
        assert len({FD(("a", "b"), "c"), FD(("b", "a"), "c")}) == 1

    def test_roundtrip_dict(self):
        fd = FD(("zip",), "city")
        assert Dependency.from_dict(fd.to_dict()) == fd


class TestCFD:
    def test_patterns_stay_aligned_after_sort(self):
        cfd = CFD(lhs=("b", "a"), rhs="c", lhs_pattern=("2", "1"), rhs_pattern="3")
        assert cfd.lhs == ("a", "b")
        assert cfd.lhs_pattern == ("1", "2")

    def test_order_invariant_equality(self):
        a = CFD(("x", "y"), "z", ("1", "2"), "3")
        b = CFD(("y", "x"), "z", ("2", "1"), "3")
        assert a == b

    def test_is_constant(self):
        assert CFD(("a",), "b", ("1",), "2").is_constant
        assert not CFD(("a",), "b", (WILDCARD,), "2").is_constant
        assert not CFD(("a",), "b", ("1",), WILDCARD).is_constant

    def test_mismatched_pattern_length_rejected(self):
        with pytest.raises(ValueError):
            CFD(("a", "b"), "c", ("1",), "2")

    def test_conflicting_patterns_rejected(self):
        with pytest.raises(ValueError):
            CFD(("a", "a"), "c", ("1", "2"), "3")

    def test_to_fd(self):
        cfd = CFD(("a",), "b", ("1",), "2")
        assert cfd.to_fd() == FD(("a",), "b")

    def test_roundtrip_dict(self):
        cfd = CFD(("a", "b"), "c", ("1", WILDCARD), "3")
        assert Dependency.from_dict(cfd.to_dict()) == cfd
