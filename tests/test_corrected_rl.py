"""Tests for the reference implementation in `corrected_rl`.

These test the estimator and the noise model against hand-computed values.
They say nothing about the archived research scripts, which are a separate
implementation; see tests/test_archive_layout.py and docs/FINDINGS.md.
"""
import unittest

import numpy as np

from corrected_rl.core import (Config, Features, Noise, SoftmaxActorCritic, evaluate,
                               n_step_target, perturb_observation, rollout, train)


class Discrete:
    """Minimal stand-in for a Gymnasium Discrete space."""

    def __init__(self, n):
        self.n = n


class Box:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)


class ToyEnv:
    """Three states in a line; action 1 pays, action 0 does not."""

    observation_space = Discrete(3)
    action_space = Discrete(2)

    def reset(self, seed=None):
        self.s = 0
        return self.s, {}

    def step(self, action):
        self.s = min(2, self.s + 1)
        return self.s, (1.0 if action == 1 else 0.0), self.s == 2, False, {}

    def close(self):
        pass


class ConstantRewardEnv:
    """Pays exactly 1 per step for `length` steps, whatever the action."""

    observation_space = Discrete(2)
    action_space = Discrete(2)

    def __init__(self, length=3):
        self.length = length

    def reset(self, seed=None):
        self.t = 0
        return 0, {}

    def step(self, action):
        self.t += 1
        return min(1, self.t), 1.0, self.t >= self.length, False, {}

    def close(self):
        pass


class NStepTargetTests(unittest.TestCase):
    """Rewards [2, 3] and values [4, 5, 6], so every number below is checkable by hand."""

    def test_one_step_bootstraps_from_the_next_state(self):
        self.assertAlmostEqual(n_step_target([2, 3], [4, 5, 6], 0, 1, 0.9), 2 + 0.9 * 5)

    def test_horizon_past_a_terminal_episode_does_not_bootstrap(self):
        """A terminal state has value zero by definition; adding a bootstrap here
        would credit the agent for a future that does not exist."""
        self.assertAlmostEqual(n_step_target([2, 3], [4, 5, 6], 0, 3, 0.9), 2 + 0.9 * 3)

    def test_truncation_does_bootstrap(self):
        """A time limit is an artefact of the run, not a property of the task, so
        the remaining value must be estimated rather than assumed to be zero."""
        self.assertAlmostEqual(n_step_target([2, 3], [4, 5, 6], 0, 4, 0.9, False),
                               2 + 0.9 * 3 + 0.81 * 6)

    def test_monte_carlo_uses_the_whole_episode(self):
        self.assertAlmostEqual(n_step_target([2, 3], [4, 5, 6], 0, None, 0.9), 2 + 0.9 * 3)

    def test_target_from_a_later_start_discounts_from_that_point(self):
        self.assertAlmostEqual(n_step_target([2, 3], [4, 5, 6], 1, 1, 0.9), 3)

    def test_empty_trajectory_is_rejected(self):
        with self.assertRaises(ValueError):
            n_step_target([], [4], 0, 1, 0.9)

    def test_mismatched_value_length_is_rejected(self):
        with self.assertRaises(ValueError):
            n_step_target([2, 3], [4, 5], 0, 1, 0.9)

    def test_start_outside_the_trajectory_is_rejected(self):
        with self.assertRaises(ValueError):
            n_step_target([2, 3], [4, 5, 6], 2, 1, 0.9)


class FeatureTests(unittest.TestCase):
    def test_discrete_features_are_one_hot_plus_a_bias(self):
        f = Features(Discrete(3))
        np.testing.assert_allclose(f(1), [0, 1, 0, 1])

    def test_discrete_observation_out_of_range_is_rejected(self):
        with self.assertRaises(ValueError):
            Features(Discrete(3))(5)

    def test_bounded_dimensions_are_scaled_to_the_unit_interval(self):
        f = Features(Box([0, 0], [10, 10]))
        np.testing.assert_allclose(f([5, 10]), [0.0, 1.0, 1.0])

    def test_unbounded_dimensions_are_squashed_instead_of_scaled(self):
        """There is no finite range to scale against, so tanh keeps the feature
        bounded rather than letting one dimension dominate the linear model."""
        f = Features(Box([0, -np.inf], [10, np.inf]))
        value = f([5, 0.5])
        self.assertAlmostEqual(value[0], 0.0)
        self.assertAlmostEqual(value[1], np.tanh(0.5))

    def test_wrong_observation_width_is_rejected(self):
        with self.assertRaises(ValueError):
            Features(Box([0, 0], [1, 1]))([1, 2, 3])


class PolicyTests(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        agent = SoftmaxActorCritic(Discrete(3), Discrete(2))
        self.assertAlmostEqual(agent.probs(0).sum(), 1.0)

    def test_softmax_is_stable_under_large_preferences(self):
        """Without subtracting the row maximum this overflows to nan."""
        agent = SoftmaxActorCritic(Discrete(2), Discrete(2))
        agent.theta[0, 0] = 1e4
        self.assertTrue(np.all(np.isfinite(agent.probs(0))))

    def test_a_positive_error_raises_the_chosen_action(self):
        agent = SoftmaxActorCritic(Discrete(3), Discrete(2))
        agent.update(0, 1, target=1.0, actor_lr=0.1, critic_lr=0.1)
        self.assertGreater(agent.probs(0)[1], 0.5)

    def test_a_negative_error_lowers_the_chosen_action(self):
        agent = SoftmaxActorCritic(Discrete(3), Discrete(2))
        agent.update(0, 1, target=-1.0, actor_lr=0.1, critic_lr=0.1)
        self.assertLess(agent.probs(0)[1], 0.5)

    def test_the_update_moves_the_critic_toward_the_target(self):
        agent = SoftmaxActorCritic(Discrete(3), Discrete(2))
        before = agent.value(0)
        agent.update(0, 0, target=5.0, actor_lr=0.01, critic_lr=0.1)
        self.assertGreater(agent.value(0), before)

    def test_continuous_action_spaces_are_rejected(self):
        with self.assertRaises(ValueError):
            SoftmaxActorCritic(Discrete(3), Box([0], [1]))


class NoiseTests(unittest.TestCase):
    def test_probabilities_outside_the_unit_interval_are_rejected(self):
        with self.assertRaises(ValueError):
            Noise(action=1.1)
        with self.assertRaises(ValueError):
            Noise(observation=-0.1)
        with self.assertRaises(ValueError):
            Noise(reward_std=-1)

    def test_discrete_observation_noise_reports_a_valid_state(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            value = perturb_observation(0, Discrete(4), 1.0, rng)
            self.assertIn(value, range(4))

    def test_zero_rate_never_perturbs(self):
        rng = np.random.default_rng(0)
        self.assertEqual(perturb_observation(2, Discrete(4), 0.0, rng), 2)

    def test_bounded_continuous_noise_stays_inside_the_bounds(self):
        rng = np.random.default_rng(0)
        value = perturb_observation(np.array([0.0, 0.0]), Box([-1, -1], [1, 1]), 1.0, rng)
        self.assertTrue(np.all(value >= -1) and np.all(value <= 1))

    def test_reward_noise_does_not_move_the_evaluation_yardstick(self):
        """The learner sees a noisy reward; the reported return must not.

        If the score were computed from the noisy reward, reward noise would
        change the measurement as well as the learning, and the experiment
        would not answer the question it asks.
        """
        env = ConstantRewardEnv(length=4)
        agent = SoftmaxActorCritic(env.observation_space, env.action_space)
        rng = np.random.default_rng(1)
        _, _, total, _ = rollout(env, agent, Noise(reward_std=50.0), rng,
                                 max_steps=10, seed=0, learning=True, gamma=0.99)
        self.assertEqual(total, 4.0)

    def test_action_noise_can_execute_an_action_the_policy_did_not_choose(self):
        """Under full action noise the executed action is independent of the
        chosen one, and the transition keeps both so the policy gradient can
        credit the action the agent believes it took."""
        env = ConstantRewardEnv(length=30)
        agent = SoftmaxActorCritic(env.observation_space, env.action_space)
        rng = np.random.default_rng(3)
        transitions, _, _, _ = rollout(env, agent, Noise(action=1.0), rng,
                                       max_steps=30, seed=0, learning=True, gamma=0.99)
        self.assertTrue(any(chosen != executed for _, chosen, executed, _, _ in transitions))


class DiscountTests(unittest.TestCase):
    def test_the_discounted_return_uses_the_configured_gamma(self):
        """A regression guard. The discount factor must come from the config
        rather than a constant written into the rollout loop, or every result
        reported at another gamma is quietly wrong.
        """
        env = ConstantRewardEnv(length=3)
        agent = SoftmaxActorCritic(env.observation_space, env.action_space)
        rng = np.random.default_rng(0)
        _, _, _, discounted = rollout(env, agent, Noise(), rng, max_steps=10,
                                      seed=0, learning=True, gamma=0.5)
        self.assertAlmostEqual(discounted, 1 + 0.5 + 0.25)

    def test_a_different_gamma_gives_a_different_discounted_return(self):
        env = ConstantRewardEnv(length=3)
        agent = SoftmaxActorCritic(env.observation_space, env.action_space)
        a = rollout(env, agent, Noise(), np.random.default_rng(0), 10, 0, True, 0.5)[3]
        b = rollout(env, agent, Noise(), np.random.default_rng(0), 10, 0, True, 0.9)[3]
        self.assertNotAlmostEqual(a, b)


class ConfigTests(unittest.TestCase):
    def test_invalid_settings_are_rejected(self):
        for kwargs in ({'n': 0}, {'gamma': 1.5}, {'actor_lr': 0}, {'episodes': 0},
                       {'max_steps': 0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                Config(**kwargs)

    def test_monte_carlo_is_expressed_as_n_none(self):
        self.assertIsNone(Config(n=None).n)


class TrainingTests(unittest.TestCase):
    def test_a_seed_fully_determines_a_run(self):
        first_agent, first = train(ToyEnv(), Config(episodes=8, seed=11))
        second_agent, second = train(ToyEnv(), Config(episodes=8, seed=11))
        np.testing.assert_allclose(first_agent.theta, second_agent.theta)
        self.assertEqual(first, second)

    def test_different_seeds_give_different_runs(self):
        a, _ = train(ToyEnv(), Config(episodes=8, seed=11))
        b, _ = train(ToyEnv(), Config(episodes=8, seed=12))
        self.assertFalse(np.allclose(a.theta, b.theta))

    def test_the_agent_learns_the_paying_action_without_noise(self):
        agent, _ = train(ToyEnv(), Config(episodes=200, seed=5, actor_lr=0.1))
        self.assertGreater(agent.probs(0)[1], 0.5)

    def test_training_runs_under_all_three_perturbations_at_once(self):
        _, history = train(ToyEnv(), Config(episodes=5), Noise(1.0, 1.0, 0.2))
        self.assertEqual(len(history), 5)

    def test_metrics_record_what_the_episode_did(self):
        _, history = train(ToyEnv(), Config(episodes=3))
        for row in history:
            self.assertEqual(set(row), {'episode', 'steps', 'raw_return',
                                        'discounted_return', 'mean_td_error', 'terminated'})
            self.assertGreaterEqual(row['steps'], 1)


class EvaluationTests(unittest.TestCase):
    def test_evaluation_returns_one_score_per_episode(self):
        agent, _ = train(ToyEnv(), Config(episodes=2))
        result = evaluate(ToyEnv(), agent, episodes=4)
        self.assertEqual(len(result['returns']), 4)
        self.assertAlmostEqual(result['mean_return'], float(np.mean(result['returns'])))

    def test_evaluation_is_clean_by_default(self):
        """Training under noise and scoring on the clean task is the usual
        question; scoring under noise answers a different one and has to be
        asked for explicitly."""
        agent, _ = train(ToyEnv(), Config(episodes=5))
        clean = evaluate(ToyEnv(), agent, episodes=6)
        self.assertEqual(clean['returns'], evaluate(ToyEnv(), agent, episodes=6)['returns'])

    def test_a_single_evaluation_episode_reports_zero_spread(self):
        agent, _ = train(ToyEnv(), Config(episodes=2))
        self.assertEqual(evaluate(ToyEnv(), agent, episodes=1)['std_return'], 0.0)


if __name__ == '__main__':
    unittest.main()
