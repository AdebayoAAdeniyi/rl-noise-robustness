"""Tests for the adapters that let the archived scripts run on current packages.

Each test corresponds to one of the three breakages documented in
experiments/compat.py and docs/ENVIRONMENT.md.
"""
import unittest

import numpy as np

try:
    import gymnasium  # noqa: F401
    from experiments.compat import (RENAMED_ENVIRONMENTS, CoordinateObservation,
                                    resolve_environment_id)
    GYMNASIUM = True
except ImportError:  # pragma: no cover
    GYMNASIUM = False


@unittest.skipUnless(GYMNASIUM, 'gymnasium is not installed')
class EnvironmentIdTests(unittest.TestCase):
    def test_the_maze_id_maps_onto_its_current_name(self):
        """Upstream renamed DangerMaze-6x6 to 5x6. Our own scripts set
        n_rows=5, n_cols=6 for that environment, so the new name describes
        the same grid we ran on."""
        self.assertEqual(resolve_environment_id('Gym-Gridworlds/DangerMaze-6x6-v0'),
                         'Gym-Gridworlds/DangerMaze-5x6-v0')

    def test_an_unchanged_id_passes_through(self):
        self.assertEqual(resolve_environment_id('Gym-Gridworlds/Penalty-3x3-v0'),
                         'Gym-Gridworlds/Penalty-3x3-v0')

    def test_a_removed_environment_raises_rather_than_guessing(self):
        """CliffWalk-5x5 no longer exists upstream. Silently substituting a
        different grid would produce results that look like the archived ones
        and are not."""
        with self.assertRaises(KeyError):
            resolve_environment_id('Gym-Gridworlds/CliffWalk-5x5-v0')

    def test_an_unrelated_id_is_left_alone(self):
        self.assertEqual(resolve_environment_id('CartPole-v1'), 'CartPole-v1')

    def test_every_archived_gridworld_id_is_accounted_for(self):
        self.assertEqual(set(RENAMED_ENVIRONMENTS), {
            'Gym-Gridworlds/Penalty-3x3-v0',
            'Gym-Gridworlds/CliffWalk-4x12-v0',
            'Gym-Gridworlds/DangerMaze-6x6-v0',
            'Gym-Gridworlds/CliffWalk-5x5-v0',
        })


if GYMNASIUM:
    class FakeEnv(gymnasium.Env):
        """A Discrete-observation env, which is what the current package returns.

        The archived scripts were written against a Box observation, so this is
        the shape that breaks them.
        """

        def __init__(self, n):
            from gymnasium import spaces
            self.observation_space = spaces.Discrete(n)
            self.action_space = spaces.Discrete(5)

        def reset(self, seed=None, options=None):
            return 0, {}

        def step(self, action):
            return 4, 0.0, False, False, {}


@unittest.skipUnless(GYMNASIUM, 'gymnasium is not installed')
class CoordinateObservationTests(unittest.TestCase):
    def test_a_flat_index_becomes_row_and_column(self):
        wrapped = CoordinateObservation(FakeEnv(9), 3, 3)
        np.testing.assert_allclose(wrapped.observation(4), [1.0, 1.0])
        np.testing.assert_allclose(wrapped.observation(0), [0.0, 0.0])
        np.testing.assert_allclose(wrapped.observation(8), [2.0, 2.0])

    def test_a_non_square_grid_indexes_by_column_count(self):
        wrapped = CoordinateObservation(FakeEnv(30), 5, 6)
        np.testing.assert_allclose(wrapped.observation(7), [1.0, 1.0])

    def test_the_observation_space_becomes_a_two_dimensional_box(self):
        """The archived scripts read observation_space.shape[0] to size their
        feature grid, which is the line that fails on a Discrete space."""
        wrapped = CoordinateObservation(FakeEnv(9), 3, 3)
        self.assertEqual(wrapped.observation_space.shape, (2,))

    def test_an_index_outside_the_grid_raises(self):
        wrapped = CoordinateObservation(FakeEnv(9), 3, 3)
        with self.assertRaises(ValueError):
            wrapped.observation(12)

    def test_degenerate_grid_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            CoordinateObservation(FakeEnv(9), 0, 3)


if __name__ == '__main__':
    unittest.main()
