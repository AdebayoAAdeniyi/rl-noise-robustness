"""Adapters that let the archived experiment scripts run on current packages.

The scripts in `gridworld/` and `classic_control/` are the code that produced
our published figures and they are kept exactly as they ran. Three things have
changed underneath them since, and each one breaks a run in a different way:

1. **Environment ids.** `Gym-Gridworlds/DangerMaze-6x6-v0` was renamed
   upstream to `DangerMaze-5x6-v0`, and `CliffWalk-5x5-v0` no longer exists at
   all. Note that our own scripts set `n_rows = 5, n_cols = 6` for the maze,
   so the 5x6 name describes the same grid we used.

2. **The `coordinate_observation` argument.** The scripts pass it to
   `gymnasium.make` to get `(row, col)` observations. Current Gym-Gridworlds
   has no such argument, and because the constructor accepts `**kwargs` it is
   swallowed silently rather than raising. The env then returns a `Discrete`
   observation and the script fails one line later on
   `env.observation_space.shape[0]` with `IndexError: tuple index out of
   range`, which says nothing about the real cause.

3. **NumPy 2.** The n-step branch assigns a size-1 array into a scalar slot
   (`returns[...] = G`). NumPy deprecated that in 1.25 and removed it in 2.0,
   so the archived scripts raise `ValueError: setting an array element with a
   sequence` on current NumPy. They ran under NumPy 1.x.

`CoordinateObservation` below addresses the second point by reproducing the
observation format the scripts expect. For the third, install `numpy<2`; see
docs/ENVIRONMENT.md for the pinned set. The first is a one-line id change.

Nothing here is used by the archived scripts themselves. It exists so that a
reader can run them, and so that the three breakages are recorded as facts
rather than discovered one traceback at a time.
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised only without gymnasium
    raise ImportError("gymnasium is required: pip install -e '.[train]'") from exc


#: Environment ids used by the archived scripts, mapped to what exists upstream now.
#: A value of None means the environment was removed and has no replacement.
RENAMED_ENVIRONMENTS = {
    'Gym-Gridworlds/Penalty-3x3-v0': 'Gym-Gridworlds/Penalty-3x3-v0',
    'Gym-Gridworlds/CliffWalk-4x12-v0': 'Gym-Gridworlds/CliffWalk-4x12-v0',
    'Gym-Gridworlds/DangerMaze-6x6-v0': 'Gym-Gridworlds/DangerMaze-5x6-v0',
    'Gym-Gridworlds/CliffWalk-5x5-v0': None,
}


class CoordinateObservation(gymnasium.ObservationWrapper):
    """Turn a flat `Discrete` grid index into the `(row, col)` Box observation.

    This restores the observation format `coordinate_observation=True` used to
    provide. The scripts build their state-aggregation features from these
    coordinates, so the format has to match or the feature grid is meaningless.
    """

    def __init__(self, env, n_rows: int, n_cols: int):
        super().__init__(env)
        if n_rows < 1 or n_cols < 1:
            raise ValueError('grid dimensions must be positive')
        self.n_rows, self.n_cols = n_rows, n_cols
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([float(n_rows - 1), float(n_cols - 1)]),
            dtype=np.float64)

    def observation(self, observation):
        index = int(np.asarray(observation).item())
        if not 0 <= index < self.n_rows * self.n_cols:
            raise ValueError(f'state index {index} is outside a {self.n_rows}x{self.n_cols} grid')
        return np.array([index // self.n_cols, index % self.n_cols], dtype=np.float64)


def resolve_environment_id(env_id: str) -> str:
    """Map an archived environment id onto one that is registered today."""
    if env_id not in RENAMED_ENVIRONMENTS:
        return env_id
    current = RENAMED_ENVIRONMENTS[env_id]
    if current is None:
        raise KeyError(
            f'{env_id} was used in our experiments but is no longer registered upstream. '
            f'Its archived figures are kept under results/figures/, but the run cannot be '
            f'repeated against the current package.')
    return current


def make_gridworld(env_id: str, n_rows: int, n_cols: int, **kwargs):
    """Build an archived gridworld with coordinate observations restored.

    Noise is configured by the environment here, exactly as in the archived
    gridworld scripts, through `random_action_prob`, `observation_noise` and
    `reward_noise_std`. The classic-control scripts apply their noise in the
    data-collection loop instead, which is a genuine difference in mechanism
    between the two halves of the study and is discussed in docs/METHOD_NOTES.md.
    """
    unsupported = {'coordinate_observation'} & set(kwargs)
    if unsupported:
        raise TypeError(
            'coordinate_observation is no longer a Gym-Gridworlds argument and is '
            'silently ignored if passed; this wrapper replaces it')
    env = gymnasium.make(resolve_environment_id(env_id), **kwargs)
    return CoordinateObservation(env, n_rows, n_cols)
