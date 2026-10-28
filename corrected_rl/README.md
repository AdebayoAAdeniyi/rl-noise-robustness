# A reference n-step actor-critic

A clean, tested implementation of the algorithm this project studied, written
so the estimator, the noise model and the update schedule are each visible in
one place.

It is a separate implementation from the archived research scripts in
[`../experiments/`](../experiments/), with a state-value critic instead of an
action-indexed one, simpler features and an episode-buffered update schedule.
It reproduces no number from the report and is not meant to. What it is for is
running a small, seeded noise experiment that someone else can follow.

```bash
python -m corrected_rl.cli --env CartPole-v1 --n 3 --episodes 300 --seed 42 \
    --action-noise 0.1
```

`core.py` holds the algorithm, `cli.py` the runner. The full list of
differences from the archived code is in
[`../docs/METHOD_NOTES.md`](../docs/METHOD_NOTES.md).
