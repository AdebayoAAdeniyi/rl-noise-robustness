# Method notes

How the experiments are built, why each choice was made, and where the two
implementations in this repository differ. Defects we found in our own code
are in [FINDINGS.md](FINDINGS.md); this document is about design.

## The three perturbations

| Noise | Mechanism | Parameter |
|---|---|---|
| Action | with probability *a*, a uniformly random action is executed; the agent records the action it chose | *a* in [0, 1] |
| Observation | with probability *o*, the agent is shown a random state; it is still in its true state | *o* in [0, 1] |
| Reward | zero-mean Gaussian noise added to the reward the learner sees | standard deviation sigma |

Two of these are probabilities and one is on the reward scale, so a noise
level of 0.5 means something different in each column of our results. Our
report states this as a limitation and it is worth repeating: the comparison
is between the *shapes* of the degradation curves, not between numbers read
off at matching parameter values.

Action noise is the interesting one. The agent chooses an action, a different
action is executed, and the agent is not told. The policy gradient therefore
credits the chosen action with the consequences of the executed one. That is
what makes action noise a learning problem rather than an exploration
schedule: an exploration bonus would attribute outcomes correctly.

## Where the noise is injected

This differs between the two halves of the study, and results should be read
with that in mind.

**Gridworlds** pass `random_action_prob`, `observation_noise` and
`reward_noise_std` to `gymnasium.make`, so the perturbations are generated
inside the environment.

**CartPole and MountainCar** apply the perturbations in `collect_data`,
because the standard Gymnasium environments have no such arguments.
Observation noise there draws uniformly inside hand-set finite bounds,
`[-4.8, -3, -0.419, -3]` to `[4.8, 3, 0.419, 3]` for CartPole, rather than
the environment's declared bounds, two of which are infinite and admit no
uniform distribution.

The mechanisms are intended to match and are implemented by different code, so
a difference between a gridworld result and a classic-control result carries a
mechanism difference alongside the environment difference.

## Features

| Environment | Features | Size |
|---|---|---|
| Gridworlds | state aggregation over a 20x20 centre grid on (row, col) | 400 |
| CartPole | RBF over a 5-per-dimension grid in 4 dimensions | 625 |
| MountainCar | RBF over a 10-per-dimension grid in 2 dimensions | 100 |

Aggregation features are one-hot in the nearest centre, which for a small
discrete grid is a tabular representation. RBFs give the continuous tasks
smooth generalisation between nearby states.

## The critic in the archived scripts

Our critic weights have shape `(features, actions)` and the value of a state
is read as `max` over the action axis:

```python
delta = G - np.dot(get_phi(s[t]), weights_v).max(-1)
weights_v[:, a[t]] += alpha_v * phi[t] * delta * fact
```

Only the column of the executed action is updated. This is closer to an
action-value function read greedily than to the state-value critic *v(S, w)*
in the pseudocode we reproduced in the report's appendix. It works, in that
the max over a greedy action-value function approximates the state value, but
it is not the algorithm as written and the report's actor-critic terminology
should be read as describing the family rather than an exact textbook
implementation.

We have not changed it. It is the code that produced the published results,
and rewriting it would mean the figures no longer correspond to the code.
`corrected_rl` implements the state-value version separately, for comparison
rather than as a replacement.

## The update scaling

The n-step branch divides each update by the number of rewards collected:

```python
weights_v[:, a[t]] += (alpha_v * get_phi(s[t]) * delta).reshape(...) * (1 / power)
weights_p += alpha_p * (gamma ** t) * delta * dlog_softmax_probs(...) * (1 / power)
```

and Monte Carlo divides by the number of remaining steps, `1 / (len(G) - t)`,
or by the episode length when `MC_update=1`. These are deliberate step-size
choices that keep the effective learning rate comparable across horizons, not
part of the standard algorithm. They matter when comparing our numbers against
a textbook implementation, because the effective step size differs.

The actor update does carry the `gamma ** t` factor, matching the `I` term in
the episodic actor-critic pseudocode. `corrected_rl` omits it, which is the
common practical variant; that is a genuine difference between the two
implementations.

## Evaluation

Expected return is measured by `expected_return()` on a separate environment
built without noise and, for the gridworlds, without reward shaping. The
policy is greedy. This is why expected return survived the logging defect
described in FINDINGS.md: it never passes through the shared metric buffers.

Two caveats on the measurement itself. `episodes_eval = 1`, so each evaluation
is a single episode. For a deterministic gridworld with a fixed start that is
exact; for CartPole and MountainCar, whose start states are random, it is a
high-variance estimate that is only smoothed by averaging over the 50 seeds.
And the gridworld evaluation env caps episodes at 30 steps, which is ample for
a 3x3 grid and tight for CliffWalk 4x12.

## The reference implementation

`corrected_rl/` is a separate, self-contained n-step actor-critic. It exists
because the archived scripts make a small experiment hard to run: settings
live in module-level globals, the active configuration is whichever line was
not commented out last, and the metric buffers are shared mutable state.

It differs from the archived code deliberately and in ways that matter:

| | Archived scripts | `corrected_rl` |
|---|---|---|
| Critic | action-indexed, max over actions | state value *V(s)* |
| Features | aggregation or RBF grids | one-hot, or scaled continuous dimensions |
| Update schedule | per step within an episode | one snapshot per episode, applied in order |
| Actor `gamma ** t` factor | present | omitted |
| Step-size scaling | divided by horizon | plain learning rate |
| Configuration | module globals | dataclasses and CLI flags |

Given those differences, scores from `corrected_rl` are not comparable with
the archived results and no claim is made that they reproduce anything. Its
value is that it is small enough to read and tested against hand-computed
targets.

### Why the study needed 50 seeds

A small demonstration with the reference implementation, 150 training episodes
on CartPole with n=3, evaluated over 20 greedy episodes:

| Action noise | Mean over 8 seeds | Standard deviation | Per-seed scores |
|---:|---:|---:|---|
| 0.0 | 207.4 | 138.1 | 169, 129, 124, 206, 288, 199, 500, 44 |
| 0.1 | 248.0 | 177.2 | 147, 334, 197, 500, 496, 18, 132, 159 |
| 0.3 | 174.4 | 148.6 | 123, 9, 151, 500, 158, 48, 192, 215 |

Seed 42 alone gives 115 without action noise and 500 at 0.3, which looks like
a large positive effect and is not one. The between-seed spread is comparable
to the mean, so eight seeds do not separate these conditions at all.

This is a demonstration of variance, not a result about action noise: the
reference implementation uses different features and a different critic from
the study, and 150 episodes is nothing. The point is how easily a single run
suggests an effect that more seeds dissolve, which is why the reported
experiments use 50 seeds and 600,000 steps per setting.

Three things in it are worth lifting into other work:

- **Bootstrapping distinguishes termination from truncation.** A terminal
  state has value zero by definition. A time-limit cutoff does not, and
  treating it as terminal tells the agent that every long episode ended in
  nothing. `n_step_target` takes a `terminal_end` flag and the tests pin both
  cases to hand-computed values.
- **Returns are accumulated from the true reward, never the noisy one.** If
  the score used the perturbed reward, reward noise would move the yardstick
  as well as the learning, and the experiment would not answer its question.
  A test asserts this with a standard deviation of 50 on a deterministic
  environment.
- **The agent and the environment draw from separate generators.** Changing
  the number of training episodes then does not shift the perturbation stream,
  so two runs differing in one setting differ in that setting alone.
