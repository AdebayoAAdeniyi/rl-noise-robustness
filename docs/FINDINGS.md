# Findings from reviewing our own code

We went back through the training scripts after the project and checked what
they actually recorded. Three things turned up. One of them affects part of
what we reported, and this document says exactly which part, with a script
that reproduces the evidence.

Putting this in the repository rather than quietly fixing it is deliberate.
The code that produced our figures is the code that is here, and a reader
should be able to see what it does and does not support.

## 1. The delta and reward histories do not hold what the episode produced

**Where:** `reinforce()`, in all four gridworld trainers, both classic-control
trainers, and four of the research variants. The two lines are:

```python
delta_error_history[tot_steps: tot_steps + ep_steps] = deltas[:ep_steps]
reward_history[tot_steps: tot_steps + ep_steps]      = returns[:ep_steps]
```

**What goes wrong:** `deltas` and `returns` are buffers of length `max_steps`,
written at offset `tot_steps + t` as each episode is processed. The slice on
the right reads from the *front* of the buffer rather than from the current
offset, so from the second episode onwards every episode writes the first
episode's values into the history. There is a second, compounding problem:
these two lines run before the current episode's updates are computed, so even
with the index corrected they would log the previous episode.

**Evidence:** run it yourself.

```bash
python verification/check_metric_logging.py experiments/gridworld/penalty.py
```

On a 400-step Penalty run the script reports:

```
  episodes logged                                   : 19
  episodes whose logged values came from that episode: 1

  delta_error: 400 entries, 10 distinct values
  reward     : 400 entries, 8 distinct values
```

Nineteen episodes produced ten distinct delta values between them, and the one
episode that matched is the first, whose logged values are zeros because
nothing had been updated yet. The script prints the slice that was written
beside the slice the episode actually produced, side by side.

**What it affects.** Four of the eight recorded metrics:

| Metric | Status |
|---|---|
| `expectation` | sound; evaluated separately by `expected_return()` on a clean env |
| `theta_stability` | sound; a scalar recomputed each iteration |
| `weights_stability` | sound; a scalar recomputed each iteration |
| `episode` | sound |
| `delta_error` | **affected** by the slice defect |
| `reward` | **affected** by the slice defect |
| `average_return` | **affected**, separately: see point 2 |
| `performance` | **affected**, separately: see point 2 |

**What this means for the report.** The main results are unaffected. Figure 1,
the normalised heatmap, and Figures 2 to 5, the final-return bar charts, are
all built from expected return, which is measured by a separate evaluation
pass on a noise-free environment and never passes through the faulty slice.
The conclusions those figures support stand: observation noise is the most
damaging perturbation, action noise hurts the gridworlds and helps the
classic-control tasks, reward noise hurts sparse-reward environments more.

The appendix is where this bites. Appendix A discusses the critic's delta
signal alongside the two variance measures. The variance measures are sound;
the **delta discussion rests on a corrupted metric** and should not be relied
on. Statements of the form "MC and n-step 10 experience higher instability in
state value", where the evidence is a delta panel, are not supported by the
data as recorded.

**Affected figures in this repository.** Of the 144 archived figures, 48 are
built from an affected metric:

| Figure family | Count | Status |
|---|---:|---|
| `results_exp_ret` curves and bar charts | 30 | sound |
| `theta_stability` curves and bar charts | 30 | sound |
| `weights_stability` curves and bar charts | 30 | sound |
| `episode` bar charts | 6 | sound |
| `delta_error` curves and bar charts | 30 | **affected** |
| `reward` bar charts | 6 | **affected** |
| `average_return` bar charts | 6 | **affected** |
| `performance` bar charts | 6 | **affected** |

The affected figures are kept rather than deleted, and the README of each
environment folder marks them.

## 2. Two running statistics are averaged over the whole preallocated buffer

**Where:** the same `reinforce()` functions:

```python
average_return = np.mean(returns)
performance_variance = np.var(returns)
```

`returns` has length `max_steps` and is filled in as the run proceeds, so
early in a run it is mostly zeros. The mean is therefore not the average
return so far; it is the sum so far divided by the total run length, which
rises roughly linearly whatever the agent does. The variance has the same
problem. Both should have been taken over `returns[:tot_steps]`.

This is why the `average_return` curves ramp. The shape is an artefact of the
denominator, not a learning curve.

## 3. The environment is reseeded identically on every episode

**Where:** `collect_data()`, which is called once per episode with
`n_episodes=1`:

```python
episode_seed = cantor_pairing(ep, seed)
s, _ = env.reset(seed=episode_seed)
```

`ep` is always 0 inside that call, so `episode_seed` is a constant for a given
run seed and `env.reset(seed=...)` restarts the environment's random stream at
the same point every episode. For the gridworlds, where the perturbations are
generated inside the environment, that means the noise stream restarts each
episode rather than continuing.

The trajectories still differ between episodes, because action selection draws
from the global NumPy generator which is not reset. So this is not a case of
every episode being identical. It does mean the perturbations an agent meets
are drawn from a repeating stream rather than an independent one, which is a
weaker form of randomisation than intended, and it is worth knowing before
treating the 50 seeds as fully independent noise realisations.

We have not quantified how much this changes the results, and we are not
claiming it does. It is recorded because it is true and checkable.

## Two design choices that look like defects and are not

**The evaluation environment uses a different reward function.** The gridworld
training env is built with `distance_reward=True` and the evaluation env
without it. That is reward shaping during training with scoring on the true
sparse task, which is the right way round. Reading the two `gymnasium.make`
calls quickly makes it look like an inconsistency.

**The policy gradient credits the chosen action, not the executed one.** Under
action noise these differ. Crediting the chosen action is what the report
describes: the agent believes it took the action it selected, so it assigns
the consequences of a different action to that one. That asymmetry is the
mechanism being studied, not an oversight. `corrected_rl` keeps it and says so
in `rollout()`.

## What we would do differently

- Log metrics from the slice the episode wrote, after the updates that
  produced them, and assert that in a test. A single check that the logged
  values differ between episodes would have caught this in minutes.
- Take running statistics over the filled part of a buffer, not the whole
  allocation.
- Advance the environment's random stream across episodes rather than
  restarting it, and record the seeding scheme next to the results.
- Separate what is measured from what is plotted. Expected return survived
  because it is computed by its own function on its own environment; the
  metrics that failed are the ones threaded through shared mutable buffers.
