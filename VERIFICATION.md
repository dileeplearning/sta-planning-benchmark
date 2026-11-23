# Release verification — 2026-11-23

Release 0.1.0 was verified in a newly created CPython 3.12.13 virtual environment
on macOS ARM64. Dependencies were downloaded into that environment, independently
of the exploratory repository's environment. Exact installed versions are pinned
in `requirements.lock`; the numerical/plotting dependencies match the original
run. The original reference data were not overwritten.

Completed checks:

1. All three unit tests passed: upstream STA recurrence equivalence, exhaustive
   max-sum/backtrace validation, and independent trajectory-scoring cases.
2. The dense upstream STA `step` body and block recurrence differed by at most
   **2.980232238769531e-7** in activity on the release test fixture (tolerance
   2e-5). This is the release fixture's result, not the historical fixture's number.
3. Integrity hashes matched for all 46 frozen data/reference/vendor files.
4. Regenerating every maze and query from seeds reproduced **all input arrays
   exactly**, including wall layouts, adjacency, distances and selected pairs.
5. The complete noiseless experiment was rerun: 3,000 queries, STA snapshots at
   400 and 1,500 updates, and exact max-sum with backtracing.
6. **Every rerun output array matched the original reference exactly**: decoded
   paths, STA peak activities and max-sum scores. All 9,000 method/query scored
   records matched; there were zero differences.
7. Rebuilding from rerun outputs produced byte-identical PNG, trial-score CSV,
   success-rate CSV and summary JSON compared with rebuilding from reference
   outputs. PDF byte equality is not asserted because PDF metadata can include
   generation timestamps. The chart was visually inspected.

Machine-readable evidence is in `results/verification/checks.json`,
`comparison.json` and `run.json`. The latter records actual library versions,
platform, thread count and per-maze planner-call timings. It is not a controlled
cross-algorithm speed benchmark.

The clean rerun itself is kept outside the distributable package in ignored
scratch space because its arrays are identical to the bundled reference arrays.
The release archive excludes the local virtual environment, Git metadata, caches
and all exploratory or interrupted experiments. These checks were completed
locally before publication.
