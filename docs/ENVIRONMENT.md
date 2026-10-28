# Running this code today

The archived experiment scripts do not run against current packages. Three
things changed underneath them after the project, and each fails differently.
They are listed here with the error each one produces, because the errors do
not point at their causes.

## 1. Two environment ids no longer exist

| Id used by our scripts | Status today |
|---|---|
| `Gym-Gridworlds/Penalty-3x3-v0` | unchanged |
| `Gym-Gridworlds/CliffWalk-4x12-v0` | unchanged |
| `Gym-Gridworlds/DangerMaze-6x6-v0` | renamed to `DangerMaze-5x6-v0` |
| `Gym-Gridworlds/CliffWalk-5x5-v0` | removed, no replacement |

The maze rename is cosmetic: our own script sets `n_rows = 5, n_cols = 6` for
that environment, so the new name describes the grid we actually ran on.

CliffWalk 5x5 is gone. Its archived figures are kept under `results/figures/`
and the run cannot be repeated against the current package.

`experiments.compat.resolve_environment_id` maps the old ids onto the current
ones and raises on the removed one rather than silently substituting a
different grid.

## 2. `coordinate_observation` is silently ignored

Our gridworld scripts call:

```python
env = gymnasium.make(env_id, coordinate_observation=True, distance_reward=True, ...)
```

Current Gym-Gridworlds has no `coordinate_observation` argument. Because its
constructor accepts `**kwargs`, the argument is swallowed without complaint
and the environment returns a `Discrete` observation instead of a `(row, col)`
Box. The script then fails on the next line:

```
state_dim = env.observation_space.shape[0]
IndexError: tuple index out of range
```

which says nothing about the real cause. `experiments.compat.CoordinateObservation`
wraps the environment and restores the observation format the scripts expect.

## 3. NumPy 2 rejects an assignment in the n-step branch

```python
returns[tot_steps + t + power] = G + (gamma**power) * ...
ValueError: setting an array element with a sequence.
```

`G` is a size-1 array, because `r = np.vstack(data['r'])` makes every reward a
one-element row. Assigning a size-1 array into a scalar slot was deprecated in
NumPy 1.25 and removed in 2.0. The scripts ran under NumPy 1.x, where it
worked. The same pattern appears four times in each trainer.

Install `numpy<2` to run them as written.

## A working set of versions

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install "numpy<2" "gymnasium>=0.29,<2" matplotlib tqdm
python -m pip install "Gym-Gridworlds"     # the gridworld environments
```

`Gym-Gridworlds` is Simone Parisi's package. It is not bundled here. Note that
the PyPI name matters: a different, unrelated package is published as
`gym_gridworlds` in lowercase.

The reference implementation under `corrected_rl/` has none of these
constraints. It needs only NumPy, plus Gymnasium if you point it at a
Gymnasium environment, and it runs on current versions:

```bash
python -m pip install -e '.[train]'
python -m unittest discover -s tests -v
```

## What runs without any of this

The tests run on current packages with no extra installs beyond NumPy and
Gymnasium. The figures under `results/` can be looked at directly. The
evidence script for the logging defect runs on current NumPy, because it
patches the four size-1 assignments in a private copy before loading:

```bash
python verification/check_metric_logging.py experiments/gridworld/penalty.py
```

## The result arrays

The `.npz` metric arrays the plotting scripts read are not in this repository.
They are large, and the figures they produced are included instead, with a
SHA-256 manifest in `results/ALL_FIGURE_PROVENANCE.json`. Point the original
plotting scripts at your own run directory with `RL_NOISE_RUN_DIR`, or use
`analysis/plot_saved_results.py`, which takes a path.
