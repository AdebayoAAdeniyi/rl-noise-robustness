"""An episodic n-step actor-critic, written to be read.

This is a clean reference implementation of the algorithm our project studied,
built so that the estimator, the noise model and the update schedule are each
visible in one place. It is deliberately separate from the archived research
scripts under `experiments/`: those are the code that produced our published
figures, and this one has different features, a different critic and a
different update schedule. It reproduces no result from the paper and is not
meant to.

What it is for is the thing the archived scripts make hard: running a small,
seeded, self-contained noise experiment that someone else can follow.

The policy is a linear softmax over state features. The critic is a linear
state-value function V(s) = phi(s) . w. The archived scripts instead carry an
action-indexed critic and take a max over actions, which is closer to an
action-value function; the difference is written up in docs/METHOD_NOTES.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Noise:
    """The three perturbations the study compares.

    `action` and `observation` are probabilities per timestep. `reward_std` is
    the standard deviation of zero-mean Gaussian noise added to the reward the
    learner sees. The three are deliberately not commensurable: two are
    probabilities and one is a reward-scale quantity, which is a limitation
    our report states rather than one this code can remove.
    """

    action: float = 0.0
    observation: float = 0.0
    reward_std: float = 0.0

    def __post_init__(self):
        if not 0 <= self.action <= 1:
            raise ValueError('action noise is a probability and must be in [0, 1]')
        if not 0 <= self.observation <= 1:
            raise ValueError('observation noise is a probability and must be in [0, 1]')
        if self.reward_std < 0:
            raise ValueError('reward noise standard deviation must be nonnegative')


@dataclass(frozen=True)
class Config:
    """Run settings. `n=None` means Monte Carlo, the full episode return."""

    n: int | None = 1
    gamma: float = 0.99
    actor_lr: float = 0.01
    critic_lr: float = 0.05
    episodes: int = 100
    max_steps: int = 500
    seed: int = 42

    def __post_init__(self):
        if self.n is not None and self.n < 1:
            raise ValueError('n must be at least 1, or None for Monte Carlo')
        if not 0 <= self.gamma <= 1:
            raise ValueError('gamma must be in [0, 1]')
        if self.actor_lr <= 0 or self.critic_lr <= 0:
            raise ValueError('learning rates must be positive')
        if self.episodes < 1 or self.max_steps < 1:
            raise ValueError('episodes and max_steps must be at least 1')


class Features:
    """Fixed state features: one-hot for discrete states, scaled for continuous.

    A constant 1 is appended in both cases so the critic has an intercept and
    the policy has a per-action bias. Without it a zero feature vector would
    force V(s) = 0 and a uniform policy at that state, which is an arbitrary
    constraint rather than a modelling choice.

    Continuous dimensions with finite bounds are mapped to [-1, 1]. Dimensions
    with infinite bounds, such as the two velocities in CartPole, are squashed
    with tanh instead, because there is no finite range to scale against.
    """

    def __init__(self, observation_space):
        self.space = observation_space
        self.is_discrete = hasattr(observation_space, 'n')
        if self.is_discrete:
            self.size = int(observation_space.n) + 1
            return
        self.low = np.asarray(observation_space.low, dtype=float).reshape(-1)
        self.high = np.asarray(observation_space.high, dtype=float).reshape(-1)
        self.bounded = np.isfinite(self.low) & np.isfinite(self.high) & (self.high > self.low)
        self.size = self.low.size + 1

    def __call__(self, observation):
        if self.is_discrete:
            index = int(np.asarray(observation).item())
            if not 0 <= index < self.space.n:
                raise ValueError(f'discrete observation {index} is outside the space')
            x = np.zeros(self.size)
            x[index] = 1.0
            x[-1] = 1.0
            return x
        obs = np.asarray(observation, dtype=float).reshape(-1)
        if obs.size != self.low.size:
            raise ValueError(f'expected {self.low.size} observation dimensions, got {obs.size}')
        scaled = np.tanh(obs)
        if self.bounded.any():
            span = self.high[self.bounded] - self.low[self.bounded]
            scaled[self.bounded] = np.clip(
                2 * (obs[self.bounded] - self.low[self.bounded]) / span - 1, -1, 1)
        return np.append(scaled, 1.0)


class SoftmaxActorCritic:
    """Linear softmax policy with a linear state-value critic.

    Both parameter sets start at zero, which makes the initial policy uniform
    and the initial value estimate zero everywhere. That is a deliberate
    choice: a seeded run is then fully determined by the seed, with no hidden
    dependence on an initialisation draw.
    """

    def __init__(self, observation_space, action_space, seed=42):
        if not hasattr(action_space, 'n'):
            raise ValueError('only discrete action spaces are supported')
        self.features = Features(observation_space)
        self.num_actions = int(action_space.n)
        self.rng = np.random.default_rng(seed)
        self.theta = np.zeros((self.features.size, self.num_actions))
        self.w = np.zeros(self.features.size)

    def probs(self, observation):
        """Action probabilities, with the row maximum subtracted before exp.

        The subtraction is free and it is the difference between training and
        an overflow once the preferences grow.
        """
        z = self.features(observation) @ self.theta
        e = np.exp(z - np.max(z))
        return e / e.sum()

    def act(self, observation, deterministic=False):
        if deterministic:
            return int(np.argmax(self.probs(observation)))
        return int(self.rng.choice(self.num_actions, p=self.probs(observation)))

    def value(self, observation):
        return float(self.features(observation) @ self.w)

    def update(self, observation, chosen_action, target, actor_lr, critic_lr):
        """One semi-gradient critic step and one policy-gradient actor step.

        Both use the same error, `target - V(s)`, computed once before either
        parameter set moves. The actor gradient for a linear softmax is
        `phi(s) (1[a = chosen] - pi(.|s))`, written here as an outer product.

        `chosen_action` is the action the policy sampled, which under action
        noise is not necessarily the action the environment executed. See
        `rollout` for why that is the right one to use here.
        """
        x = self.features(observation)
        error = float(target - x @ self.w)
        advantage_direction = -self.probs(observation)
        advantage_direction[chosen_action] += 1.0
        self.w += critic_lr * error * x
        self.theta += actor_lr * error * np.outer(x, advantage_direction)
        return error


def n_step_target(rewards, values, start, n, gamma, terminal_end=True):
    """The n-step return from timestep `start`, or the full return when n is None.

    `values` has one more entry than `rewards`: `values[t]` is V(S_t) for every
    visited state, and `values[-1]` is V of the state the episode ended in,
    which is used only when the episode was cut off rather than finished.

    The bootstrap rule is the part worth getting right. If the horizon lands
    inside the trajectory, bootstrap from the state at that index. If it runs
    past the end, there are two cases that look alike and are not: a genuine
    termination has value zero by definition and must not be bootstrapped,
    while a time-limit truncation is an artefact of the run and should be,
    otherwise every long episode is told its future was worth nothing.
    """
    horizon = len(rewards)
    if horizon == 0:
        raise ValueError('cannot form a target from an empty trajectory')
    if len(values) != horizon + 1:
        raise ValueError('values must hold one entry per state plus the final state')
    if not 0 <= start < horizon:
        raise ValueError('start index is outside the trajectory')
    if n is not None and n < 1:
        raise ValueError('n must be at least 1, or None for Monte Carlo')

    stop = horizon if n is None else min(horizon, start + n)
    total = sum(gamma ** (j - start) * float(rewards[j]) for j in range(start, stop))
    if stop < horizon or not terminal_end:
        total += gamma ** (stop - start) * float(values[stop])
    return total


def perturb_observation(observation, space, rate, rng):
    """With probability `rate`, report a different state than the true one.

    For a discrete space this is a uniform draw over the states, which matches
    the report's description: the agent is still in its true state and only
    the observation is wrong. For a bounded continuous space it is a uniform
    draw inside the bounds. Unbounded dimensions have no uniform distribution
    to draw from, so the observation is displaced by standard Gaussian noise
    instead. That last case is a substitution, not the same mechanism, and it
    is why CartPole results from this module are not comparable with the
    archived CartPole runs, which drew uniformly inside hand-set finite bounds.
    """
    if not rate or rng.random() >= rate:
        return observation
    if hasattr(space, 'n'):
        return int(rng.integers(space.n))
    low = np.asarray(space.low, dtype=float)
    high = np.asarray(space.high, dtype=float)
    if not np.all(np.isfinite(low) & np.isfinite(high)):
        return np.asarray(observation) + rng.normal(size=np.asarray(observation).shape)
    return rng.uniform(low, high)


def rollout(env, agent, noise, rng, max_steps, seed, learning, gamma):
    """Run one episode and return its transitions and returns.

    Action noise replaces the executed action with a uniform random one, while
    the transition records the action the policy actually chose. That asymmetry
    is the mechanism the report describes: the agent believes it took the
    action it selected, so the policy gradient credits that action with the
    consequences of a different one. It is what makes action noise a learning
    problem rather than just an exploration schedule.

    Returns are accumulated from the true environment reward, never the noisy
    one, so that reward noise degrades learning without also moving the yardstick.
    """
    observation, _ = env.reset(seed=seed)
    observation = perturb_observation(observation, env.observation_space, noise.observation, rng)
    transitions = []
    total_return = 0.0
    discounted_return = 0.0
    discount = 1.0
    terminated = False

    for _ in range(max_steps):
        chosen = agent.act(observation, deterministic=not learning)
        executed = int(rng.integers(agent.num_actions)) if rng.random() < noise.action else chosen
        next_observation, reward, term, trunc, _ = env.step(executed)
        learner_reward = float(reward) + float(rng.normal(0, noise.reward_std))
        next_observation = perturb_observation(
            next_observation, env.observation_space, noise.observation, rng)

        transitions.append((observation, chosen, executed, learner_reward, next_observation))
        total_return += float(reward)
        discounted_return += discount * float(reward)
        discount *= gamma
        observation = next_observation
        terminated = bool(term)
        if term or trunc:
            break

    return transitions, terminated, total_return, discounted_return


def train(env, config=Config(), noise=Noise()):
    """Train for `config.episodes` episodes and return the agent and its metrics.

    Each episode is collected in full, the state values are snapshotted, every
    n-step target is formed against that snapshot, and the updates are then
    applied in order. This is an episode-buffered schedule, not the online
    schedule of the textbook algorithm, and not the schedule the archived
    scripts use either. It is chosen here because a single snapshot makes the
    targets within an episode mutually consistent and the run easier to reason
    about; the cost is that the targets are slightly stale by the end of a long
    episode.

    The environment and the agent draw from separate generators, so changing
    the number of training episodes does not alter the perturbation stream.
    """
    rng = np.random.default_rng(config.seed + 1000)
    agent = SoftmaxActorCritic(env.observation_space, env.action_space, config.seed)
    metrics = []

    for episode in range(config.episodes):
        transitions, terminated, total_return, discounted_return = rollout(
            env, agent, noise, rng, config.max_steps, config.seed + episode, True, config.gamma)
        rewards = [step[3] for step in transitions]
        values = [agent.value(step[0]) for step in transitions]
        values.append(agent.value(transitions[-1][4]))

        errors = []
        for t, (observation, chosen, _, _, _) in enumerate(transitions):
            target = n_step_target(rewards, values, t, config.n, config.gamma,
                                   terminal_end=terminated)
            errors.append(agent.update(observation, chosen, target,
                                       config.actor_lr, config.critic_lr))

        metrics.append({
            'episode': episode + 1,
            'steps': len(transitions),
            'raw_return': total_return,
            'discounted_return': discounted_return,
            'mean_td_error': float(np.mean(errors)),
            'terminated': terminated,
        })

    return agent, metrics


def evaluate(env, agent, episodes=20, seed=100_000, max_steps=500, gamma=0.99, noise=Noise()):
    """Score the greedy policy over fresh episodes, on true environment reward.

    Evaluation seeds are deliberately far from the training seeds so that the
    agent is not scored on episodes it was trained on. Noise defaults to none:
    the usual question is how well an agent trained under noise performs on the
    clean task, and passing a `noise` argument asks the other question instead.
    """
    rng = np.random.default_rng(seed + 1000)
    scores = []
    for i in range(episodes):
        _, _, score, _ = rollout(env, agent, noise, rng, max_steps, seed + i, False, gamma)
        scores.append(score)
    return {
        'episodes': episodes,
        'mean_return': float(np.mean(scores)),
        'std_return': float(np.std(scores, ddof=1)) if episodes > 1 else 0.0,
        'returns': scores,
    }
