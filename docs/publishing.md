# Publishing to PyPI

Releases are built and published by `.github/workflows/wheels.yml` when a
`v*` tag is pushed. PyPI upload is gated behind the `PYPI_PUBLISH` repository
variable so accidental tags never publish.

## One-time setup (maintainer)

1. Create the `nesydep` project on PyPI (or use "pending publisher" without
   creating it first):
   - PyPI → *Your projects* → *Publishing* → *Add a new pending publisher*
   - Owner: `ruochunjin`, repository: `NeSyDep`,
     workflow: `wheels.yml`, environment: `pypi`
2. In the GitHub repo: *Settings → Environments* → create environment `pypi`.
3. *Settings → Secrets and variables → Actions → Variables* → add
   `PYPI_PUBLISH = true`.

## Releasing

```bash
# bump version in pyproject.toml and src/nesydep/__init__.py, update CHANGELOG
git commit -am "vX.Y.Z"
git tag -a vX.Y.Z -m "NeSyDep vX.Y.Z"
git push origin main vX.Y.Z
```

The tag triggers cibuildwheel (Linux/macOS/Windows wheels), a clean-environment
golden regression against the built wheels, then PyPI upload via trusted
publishing (no tokens stored).

## Checklist before v1.0.0

- [ ] CI green on main (lint + 3 OS × 4 Pythons)
- [ ] Integration workflow green (`tests/integration`, slow tests)
- [ ] Benchmarks regenerated (`python scripts/benchmark.py`)
- [ ] CHANGELOG updated
- [ ] Docs build (`mkdocs build`)
