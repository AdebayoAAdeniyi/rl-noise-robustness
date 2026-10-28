"""Evidence for the metric-logging defect described in docs/FINDINGS.md.

Run this and it shows, from the archived code itself, that the `delta_error`
and `reward` histories written by `reinforce()` do not hold the values the
episode produced. Every episode after the first writes the *first* episode's
values, because the slice is read from the front of the buffer:

    delta_error_history[tot_steps : tot_steps + ep_steps] = deltas[:ep_steps]

rather than from the current offset. The logging also runs before the episode's
updates are computed, so even the index-corrected version would be one episode
behind.

Usage:
    python verification/check_metric_logging.py experiments/gridworld/penalty.py

The script patches two things in a private copy of the trainer and neither one
touches the logging logic under test:

* size-1 array assignments, which NumPy 2 rejects (see experiments/compat.py);
* a probe that records, per episode, both the slice that was written and the
  slice that the episode actually produced.

The trainer, its update rule and its logging lines are otherwise untouched.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.compat import CoordinateObservation, resolve_environment_id  # noqa: E402

NUMPY2_FIXES = [
    ("                deltas[tot_steps + t] = delta\n"
     "                returns[tot_steps + t] = G[t]",
     "                deltas[tot_steps + t] = float(np.ravel(delta)[0])\n"
     "                returns[tot_steps + t] = float(np.ravel(G[t])[0])"),
    ("                    returns[tot_steps + t + power] = G + (gamma**power) * "
     "np.dot(get_phi(s[t + power]), weights_v).max(-1).item()",
     "                    returns[tot_steps + t + power] = float(np.ravel(G)[0] + "
     "(gamma**power) * np.dot(get_phi(s[t + power]), weights_v).max(-1).item())"),
    ("                    deltas[tot_steps + t +power] = delta",
     "                    deltas[tot_steps + t +power] = float(np.ravel(delta)[0])"),
    ("                    deltas[tot_steps + t] = delta\n"
     "                    returns[tot_steps + t] = G",
     "                    deltas[tot_steps + t] = float(np.ravel(delta)[0])\n"
     "                    returns[tot_steps + t] = float(np.ravel(G)[0])"),
]

PROBE_ANCHOR = "        delta_error_history[tot_steps: tot_steps + ep_steps] = deltas[:ep_steps]"
PROBE = ("        _PROBE.append((tot_steps, ep_steps, deltas[:ep_steps].copy(),\n"
         "                       deltas[tot_steps: tot_steps + ep_steps].copy()))\n"
         + PROBE_ANCHOR)


def instrumented_copy(source: Path) -> Path:
    text = source.read_text()
    for old, new in NUMPY2_FIXES:
        if old in text:
            text = text.replace(old, new)
    if PROBE_ANCHOR not in text:
        raise SystemExit(f'{source} does not contain the logging line under test')
    text = text.replace(PROBE_ANCHOR, PROBE)
    text = text.replace('np.set_printoptions(precision=3, suppress=True)',
                        'np.set_printoptions(precision=3, suppress=True)\n_PROBE = []')
    target = Path(tempfile.mkdtemp()) / source.name
    target.write_text(text)
    return target


def load(path: Path):
    spec = importlib.util.spec_from_file_location('archived_trainer', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(module, env_id, rows, cols, max_steps, episode_cap):
    """Populate the globals the archived `reinforce()` reads, as its __main__ does."""
    import gymnasium

    resolved = resolve_environment_id(env_id)
    module.env = CoordinateObservation(
        gymnasium.make(resolved, distance_reward=True, max_episode_steps=episode_cap),
        rows, cols)
    module.env_eval = CoordinateObservation(
        gymnasium.make(resolved, max_episode_steps=30), rows, cols)
    module.episodes_eval = 1
    state_dim = module.env.observation_space.shape[0]
    module.n_actions = module.env.action_space.n
    centers_per_dim = [20] * state_dim
    low = module.env.observation_space.low
    high = module.env.observation_space.high
    module_centers = np.array(np.meshgrid(*[
        np.linspace(low[i] - (high[i] - low[i]) / centers_per_dim[i] * 0.1,
                    high[i] + (high[i] - low[i]) / centers_per_dim[i] * 0.1,
                    centers_per_dim[i])
        for i in range(state_dim)])).reshape(state_dim, -1).T
    module.get_phi = lambda s: module.aggregation_features(
        np.asarray(s, dtype=float).reshape(-1, state_dim), module_centers)
    module.phi_dummy = module.get_phi(module.env.reset()[0])
    module.gamma = 0.99
    module.alpha_p = module.alpha_v = 0.001
    module.max_steps = max_steps
    module.n_rows, module.n_cols = rows, cols


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('script', type=Path, help='an archived gridworld trainer')
    parser.add_argument('--env-id', default='Gym-Gridworlds/Penalty-3x3-v0')
    parser.add_argument('--rows', type=int, default=3)
    parser.add_argument('--cols', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=400,
                        help='total environment steps; small by design, the defect '
                             'appears from the second episode onwards')
    parser.add_argument('--episode-cap', type=int, default=25)
    parser.add_argument('--n', type=int, default=1)
    args = parser.parse_args(argv)

    module = load(instrumented_copy(args.script))
    configure(module, args.env_id, args.rows, args.cols, args.max_steps, args.episode_cap)
    np.random.seed(0)
    metrics = module.reinforce(n=args.n, MC_update=0, seed=0)

    probe = module._PROBE
    matching = sum(np.allclose(written, produced) for _, _, written, produced in probe)
    print('\n' + '=' * 72)
    print('  episode |  values written to the history  |  values the episode produced')
    print('=' * 72)
    for i, (_, _, written, produced) in enumerate(probe[:10], start=1):
        print(f'  {i:7d} |  {np.round(written[:4], 3)}  |  {np.round(produced[:4], 3)}')
    print('=' * 72)
    print(f'  episodes logged                                   : {len(probe)}')
    print(f'  episodes whose logged values came from that episode: {matching}')

    delta, reward = metrics['delta_error'], metrics['reward']
    print(f'\n  delta_error: {delta.size} entries, {np.unique(delta).size} distinct values')
    print(f'  reward     : {reward.size} entries, {np.unique(reward).size} distinct values')
    print('\n  Metrics computed per iteration rather than through that slice are unaffected:')
    for key in ('expectation', 'theta_stability', 'weights_stability'):
        print(f'    {key:20s} {np.unique(metrics[key]).size:5d} distinct values')
    return 0 if matching < len(probe) else 1


if __name__ == '__main__':
    raise SystemExit(main())
