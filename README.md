# What hinders reinforcement learning more: action, observation, or reward noise?

An empirical study of how three kinds of environmental noise degrade n-step
actor-critic learning, across four environments, five update horizons and 50
seeds per setting.

Joint work with Isaiah Heidt, University of Alberta, CMPUT 655. The report is
in [`docs/historical_reports/`](docs/historical_reports/), the training code
that produced the results is in [`experiments/`](experiments/), and the
figures are in [`results/`](results/).

**The question.** Real deployments are noisy in several different ways at
once: actuators slip, sensors lie, and reward signals are inconsistent.
Those are usually lumped together as "noise". We asked whether they are
actually interchangeable, and found that they are not.

## What we found

**Observation noise is the most damaging of the three.** It degrades
performance in every environment we tested, at levels as low as 0.01 in the
gridworlds. If the agent does not know where it is, nothing downstream can
recover.

**Action noise depends entirely on how risky the environment is.** It wrecked
the hazard-dense gridworlds, where one random step can drop the agent into a
penalty state that is reachable from almost anywhere. In CartPole and
MountainCar it *helped* at some levels, because a single bad action there is
recoverable and the randomness buys exploration. The same perturbation, the
same magnitude, opposite sign of effect.

**Reward noise tracks reward sparsity.** It hurt the gridworlds, where the
only non-zero signals come from penalty and goal states, and had little effect
on CartPole and MountainCar, where every transition pays. Gaussian noise does
not change the expectation of a return, so an agent that sees many rewards can
average through it; an agent that sees three can not.

**The update horizon interacts with the noise type.** Longer horizons were
more robust to observation noise, because one corrupted observation is diluted
across a larger sample of returns. Under action noise in the risky gridworlds,
Monte Carlo held up better, while shorter horizons did better in the gentler
control tasks. There is no horizon that is simply more robust.

**Some configurations could not learn at all,** even without noise: n=10 in
DangerMaze, and Monte Carlo in MountainCar, where episodes are long enough
that waiting for termination leaves almost no updates.

### Setup

| | |
|---|---|
| Environments | Penalty 3x3, DangerMaze, CartPole v1, MountainCar v0 |
| Algorithms | n-step actor-critic, n = 1, 2, 3, 10, and Monte Carlo |
| Scale | 50 seeds and 600,000 environment steps per setting |
| Metrics | expected return, critic error, actor and critic parameter variance |
| Compute | Digital Research Alliance of Canada |

Two further gridworlds, CliffWalk 4x12 and CliffWalk 5x5, were run and their
figures are included, but they are not part of the report's four-environment
comparison and no conclusion here rests on them.

## We reviewed our own code afterwards, and found a defect

Going back through the training scripts, we found that the metric logging does
not record what the episodes produced. Two lines read their slice from the
front of the buffer instead of the current offset, so from the second episode
onwards each episode writes the first episode's values into the history.

This is written up with reproducible evidence in
[`docs/FINDINGS.md`](docs/FINDINGS.md), and you can run the check:

```bash
python verification/check_metric_logging.py experiments/gridworld/penalty.py
```

On a short Penalty run, 19 episodes produce 10 distinct delta values between
them, where they should produce hundreds.

**What it does and does not touch.** Expected return is measured by a separate
evaluation pass on a clean environment and never goes through those buffers,
so the headline results above, and Figures 1 to 6 of the report, are
unaffected. The parameter-variance metrics are also sound. What is affected is
the critic delta signal, which the report's Appendix A discusses, along with
three running statistics. Of the 144 archived figures, 96 are sound and 48
rest on an affected metric. FINDINGS.md names which.

We have left the original scripts untouched rather than quietly repairing
them, because they are the code the published figures came from. The fix, and
the test that would have caught it, are described there.

## Repository layout

```text
experiments/
  gridworld/           Penalty, DangerMaze, CliffWalk trainers as they ran
  classic_control/     CartPole and MountainCar trainers as they ran
  research_variants/   serial and parallel savers, reruns, exploratory scripts
  compat.py            adapters that let the archived scripts run today
corrected_rl/          a clean, tested reference n-step actor-critic
analysis/
  original_plot_scripts/   our aggregation, plotting and heatmap scripts
  plot_saved_results.py    portable plotter for the saved .npz schema
results/figures/       144 archived figures with a SHA-256 provenance manifest
verification/          the script that reproduces the logging defect
tests/                 52 tests
docs/
  FINDINGS.md          what we found reviewing our own code, with evidence
  METHOD_NOTES.md      design decisions, and how the two implementations differ
  ENVIRONMENT.md       why the archived scripts need pinned versions, and which
  historical_reports/  our submitted report
```

## Running it

The archived scripts need pinned versions, for three specific reasons set out
in [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md): two environment ids were
renamed or removed upstream, the `coordinate_observation` argument is now
silently ignored, and NumPy 2 rejects a size-1 array assignment in the n-step
branch.

The reference implementation has no such constraints:

```bash
python -m pip install -e '.[train]'
python -m unittest discover -s tests -v          # 52 tests

python -m corrected_rl.cli --env CartPole-v1 --n 3 --episodes 300 --seed 42
python -m corrected_rl.cli --env CartPole-v1 --n 3 --episodes 300 --seed 42 \
    --action-noise 0.1
python -m corrected_rl.cli --env CartPole-v1 --n mc --episodes 300 --seed 42 \
    --observation-noise 0.05
```

Each run writes a per-episode CSV and a summary JSON carrying the full
configuration, so a result file says what produced it.

`corrected_rl` is a separate implementation with a state-value critic,
different features and a different update schedule. It reproduces no number
from the report and is not meant to; it exists so the algorithm can be read
and a small noise experiment run in seconds. The differences are tabulated in
[`docs/METHOD_NOTES.md`](docs/METHOD_NOTES.md).

## Points worth taking from this

- **Distinguish termination from truncation when you bootstrap.** A terminal
  state is worth zero by definition. A time-limit cutoff is not, and treating
  it as terminal teaches the agent that every long episode ended in nothing.
  `n_step_target` takes an explicit flag and both cases are pinned to
  hand-computed values in the tests.
- **Score on the true reward, never the perturbed one.** If reward noise moves
  the measurement as well as the learning, the experiment stops answering its
  question. A test asserts this with a noise standard deviation of 50.
- **Under action noise, be explicit about which action the gradient credits.**
  The agent chose one action and a different one executed. Crediting the
  chosen action is what makes this a learning problem rather than an
  exploration bonus, and it should be a documented decision rather than
  whichever variable was in scope.
- **Give the agent and the environment separate random generators.** Otherwise
  changing the episode count also changes the noise stream, and two runs
  differ in more than the setting under test.
- **Assert that a logged metric varies.** One test checking that consecutive
  episodes log different values would have caught the defect above in minutes.
  Metrics threaded through shared mutable buffers are where this goes wrong;
  the one metric here that was computed by its own function on its own
  environment is the one that survived.
- **Noise types are not a single axis.** Perturbations that are equally sized
  on paper can differ in sign of effect, and the thing that predicts which is
  a property of the environment, how costly a single mistake is, and how
  often reward arrives.

## Scope and limitations

These are the limitations our report states, and they still hold. There was no
hyperparameter tuning, so a configuration that learns poorly may be badly
tuned rather than badly suited. The environments are small and discrete-action.
The three noise types are not on a common scale: two are probabilities and one
is a reward-scale standard deviation, so the comparison is between the shapes
of the degradation curves rather than between values at matching parameters.
Parameter variance within a weight vector is a diagnostic, not a convergence
test. And the noise is injected by the environment in the gridworlds and by
the collection loop in the classic-control tasks, so those two halves differ
in mechanism as well as in task.

## Attribution

The report is joint work with Isaiah Heidt. We thank Simone Parisi for
guidance throughout the project, and the Digital Research Alliance of Canada
for the compute.

The gridworld environments come from Simone Parisi's Gym-Gridworlds package
and are not bundled here. The classic-control environments are Gymnasium's.
Neither is our work.
