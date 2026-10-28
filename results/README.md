# Figures

144 figures from the project, grouped by environment. Each one is an original
output from our runs; nothing here was recalculated when the repository was
assembled.

| Environment | Figures | Role |
|---|---:|---|
| [Penalty-3x3-v0](figures/Penalty-3x3-v0/) | 24 | main report |
| [DangerMaze-6x6-v0](figures/DangerMaze-6x6-v0/) | 24 | main report |
| [CartPole-v1](figures/CartPole-v1/) | 24 | main report |
| [MountainCar-v0](figures/MountainCar-v0/) | 24 | main report |
| [CliffWalk-4x12-v0](figures/CliffWalk-4x12-v0/) | 24 | additional, not in the report |
| [CliffWalk-5x5-v0](figures/CliffWalk-5x5-v0/) | 24 | additional, not in the report |

Each filename carries its settings: environment, metric, seed count, learning
rate, and the three noise levels in the order reward, action, observation.
`ALL_FIGURE_PROVENANCE.json` records every file's original path in the run
archive along with its byte count and SHA-256 digest, and
`tests/test_archive_layout.py` verifies each digest.

**48 of the 144 are built from a metric affected by the logging defect** in
[`../docs/FINDINGS.md`](../docs/FINDINGS.md): the `delta_error`, `reward`,
`average_return` and `performance` families. They are kept rather than
deleted, and each environment's README lists which of its files are affected.
The 96 figures covering expected return, actor parameter variance, critic
weight variance and episode length are sound.

The `.npz` metric arrays these plots were drawn from are not in the
repository. They are large, and the figures are what the report used.
