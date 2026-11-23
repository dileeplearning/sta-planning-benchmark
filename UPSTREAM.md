# Attribution and source provenance

Original repository: https://github.com/KrisJensen/pysta

Pinned commit: `fdd6dd39df54a67aa451a24d3adc280f25136601`

The files `vendor/pysta/agents.py` and `vendor/pysta/maze_utils.py` are unmodified
snapshots from that commit. Their original license and copyright notice appear
in `vendor/pysta/LICENSE`. SHA-256 hashes are in `MANIFEST.json`.

- `maze_utils.py` is loaded directly only for optional input regeneration. Its
  maze generator and adjacency construction produce the frozen experiment inputs.
- The original `SpaceTimeAttractor.step` body in `agents.py` is extracted by Python
  AST for the numerical equivalence test. We supply a small dense-weight harness
  with already-prepared inputs. The full upstream package and plotting stack are
  not imported, and the test does not claim end-to-end environment equivalence.
- `benchmark/core.py` adapts the STA recurrence into time-neighbor blocks. The
  max-sum implementation, benchmark orchestration, evaluation and publication
  materials are additions from this independent analysis.

The original exploratory run also performed an upstream-class equivalence check
with the actual upstream environment on a small maze. Its recorded discrepancy
is retained in `results/reference/main_config.json`. The release test uses a
different dense harness and reports its own discrepancy, rather than reusing
that historical number as though the test were identical.

Original reference arrays were extracted from the local experiment's
`main_maze*.npz` (noiseless STA subset) and `maxsum_maze*.npz` files without changing
their values. `results/reference/PROVENANCE.json` records hashes of those source
containers. Arrays for unrelated noisy or exploratory conditions were omitted.
The included original configuration records the broader source run; METHODS.md
defines the noiseless subset presented here.

No modifications from the exploratory clone's path/animation utilities are
needed by this standalone package. No upstream weights or training data are
downloaded at run time. Please cite the original paper and repository in addition
to this independent benchmark when using these results.
