# CartPole and MountainCar

The gentler half of the study: continuous state, discrete actions, and no
state that ends an episode in one bad step. This is where action noise stopped
hurting and started helping, which is the sharpest contrast in our results.

Both tasks use RBF features rather than the state aggregation the gridworlds
use, 625 centres for CartPole and 100 for MountainCar.

Noise is applied inside `collect_data` here, not by the environment, because
the standard Gymnasium environments take no noise arguments. Observation noise
draws uniformly inside hand-set finite bounds, since two of CartPole's
declared bounds are infinite and admit no uniform distribution. That is a
genuine difference in mechanism from the gridworld scripts and is discussed in
[`../../docs/METHOD_NOTES.md`](../../docs/METHOD_NOTES.md).

Monte Carlo struggled badly in MountainCar: episodes run long enough that
waiting for termination leaves very few updates inside the step budget.

Development versions, reruns and the combined cross-environment runners are in
[`../research_variants/`](../research_variants/).
