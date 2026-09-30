"""Regenerate golden reference outputs from the *original prototype code*.

Golden outputs are the safety net for the whole refactor: every rewritten
component must reproduce them (as set-level comparisons) on the frozen inputs
in ``tests/data/``.

Why a script instead of committed outputs?
------------------------------------------
The prototype miners are C++/MPI programs; the machine that froze the test
*datasets* had no C++ toolchain. Golden outputs must therefore be generated
once on a machine that can build the prototypes (Linux with OpenMPI is the
path of least resistance), then committed under ``tests/regression/golden/``.

Procedure
---------
1. BSFD (FD mining) — repo ``code-for-BSFD``:

   - Build the parallel miner::

       cd "code-for-BSFD/parallel discovery" && cmake -B build && cmake --build build

   - Prepare ``input.txt`` pointing at copies of
     ``tests/data/german_credit_sample.csv`` and ``tests/data/alarm_sample.csv``.
   - Run ``mpiexec -n 4 ./build/BSFD`` and save the FD output as
     ``golden/bsfd_<dataset>.txt``.

   - Baseline oracle: build ``code-for-BSFD/baselines/mainE.cpp`` (TANE/DFD are
     header-only, no MPI needed) and run TANE on the same inputs; save as
     ``golden/tane_<dataset>.txt``. TANE is a well-established correct
     implementation and serves as the FD oracle in regression tests.

2. SCFDM (CFD mining) — repo ``code-for-CFD``:

   - Build ``parallel/SCFDM_all`` (CMake).
   - Use ``tests/data/census42_sample.csv`` (excerpt of the bundled
     CENSUS42-10000) and ``tests/data/alarm_sample.csv`` as sub-tables; a
     minimal ``input.txt`` is::

       census42_sample.csv
       5            # support (tuned for the 500-row sample)
       1            # confidence
       3            # max LHS size
       FD-First-DFS-dfs

   - Save outputs as ``golden/scfdm_<dataset>.txt``. Run once with strategy
     ``Integrated-BFS`` too — that is CTane, the CFD baseline oracle
     (``golden/ctane_<dataset>.txt``).

3. Python-side stages (sampling, correlation extraction) are stochastic; freeze
   their *outputs* with ``PYTHONHASHSEED=0`` and fixed ``random_state`` /
   ``seed`` parameters where the prototype exposes them. When it does not,
   freeze the artifact (sample CSV, correlated sets) rather than re-deriving it.

Comparison rules
----------------
- FDs: compare as sets of ``(tuple(sorted(lhs)), rhs)`` pairs.
- CFDs: normalise before comparing (sort LHS attributes together with their
  pattern values; treat ``_`` as the wildcard token) — reuse the normalisation
  in ``nesydep.io.cfd_text``.
"""
