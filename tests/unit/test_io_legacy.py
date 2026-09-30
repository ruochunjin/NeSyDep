"""Round-trip tests for the legacy text formats."""
from nesydep.core.dependency import CFD, FD
from nesydep.io.legacy import load_dependencies, parse_cfd, parse_fd, parse_line


def test_parse_fd():
    assert parse_fd("[zip, city] -> state") == FD(("zip", "city"), "state")


def test_parse_cfd_constant():
    cfd = parse_cfd("[city, zip] => state, (New York, 10001 || NY)")
    assert cfd == CFD(("city", "zip"), "state", ("New York", "10001"), "NY")


def test_parse_cfd_wildcard():
    cfd = parse_cfd("[zip] => city, (_ || _)")
    assert cfd is not None and not cfd.is_constant


def test_parse_line_prefers_cfd():
    dep = parse_line("[a] => b, (1 || 2)")
    assert isinstance(dep, CFD)
    dep = parse_line("[a] -> b")
    assert isinstance(dep, FD)


def test_parse_garbage_returns_none():
    assert parse_line("this is not a dependency") is None


def test_str_roundtrip(tmp_path):
    deps = [
        FD(("zip",), "city"),
        CFD(("city",), "state", ("Chicago",), "IL"),
        CFD(("zip",), "city", ("_",), "_"),
    ]
    out = tmp_path / "deps.txt"
    from nesydep.io.legacy import dump_dependencies

    dump_dependencies(deps, out)
    loaded = load_dependencies(out)
    assert set(loaded) == set(deps)


def test_load_skips_blank_lines(tmp_path):
    p = tmp_path / "deps.txt"
    p.write_text("[a] -> b\n\n\n[c] -> d\n", encoding="utf-8")
    assert len(load_dependencies(p)) == 2
