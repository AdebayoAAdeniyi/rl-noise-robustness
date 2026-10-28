# Gridworld experiments

The gridworlds are the risky half of the study. Penalty states are reachable
from nearly every cell, so a single random action can cost an episode, and
reward is sparse: the only non-zero signals come from penalty and goal tiles.
That combination is what makes action noise and reward noise bite here while
leaving CartPole and MountainCar comparatively unharmed.

| Script | Environment | Role |
|---|---|---|
| `penalty.py` | `Gym-Gridworlds/Penalty-3x3-v0` | main report |
| `dangermaze.py` | `Gym-Gridworlds/DangerMaze-6x6-v0` | main report |
| `cliffwalk_4.py` | `Gym-Gridworlds/CliffWalk-4x12-v0` | additional, not in the report |
| `cliffwalk_5.py` | `Gym-Gridworlds/CliffWalk-5x5-v0` | additional, not in the report |

Each trainer sweeps n in {1, 2, 3, 10} plus Monte Carlo over a grid of noise
settings, 50 seeds in parallel, 600,000 environment steps per setting. Noise
is configured on the environment here through `random_action_prob`,
`observation_noise` and `reward_noise_std`, which is a different mechanism
from the classic-control scripts.

The report's conclusions cover Penalty and DangerMaze. CliffWalk was run and
its figures are kept, but nothing in the report rests on it.

Two things to know before running these:

- The active configuration is whichever `noise_combos` and `env_ids`
  assignment comes last in `__main__`. Earlier lines declare the full grid and
  are then overwritten, so the script as committed runs one setting rather
  than the whole sweep.
- They need pinned versions and, for DangerMaze, an id change. See
  [`../../docs/ENVIRONMENT.md`](../../docs/ENVIRONMENT.md) and
  [`../compat.py`](../compat.py).
