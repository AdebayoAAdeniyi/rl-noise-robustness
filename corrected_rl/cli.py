"""Command-line runner for a single seeded noise experiment.

One run writes a per-episode CSV and a summary JSON carrying the full
configuration, so a result file is self-describing and can be matched back to
the command that produced it.

This runs the reference implementation in `core.py`, not the archived research
scripts. It is a small, fast experiment, not a re-run of the 50-seed study.
"""
import argparse
import csv
import json
from pathlib import Path

from .core import Config, Noise, evaluate, train


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', default='CartPole-v1',
                        help='any Gymnasium environment id with a discrete action space')
    parser.add_argument('--n', default='1',
                        help="update horizon: an integer, or 'mc' for Monte Carlo")
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--actor-lr', type=float, default=0.01)
    parser.add_argument('--critic-lr', type=float, default=0.05)
    parser.add_argument('--action-noise', type=float, default=0.0,
                        help='probability that a uniformly random action is executed')
    parser.add_argument('--observation-noise', type=float, default=0.0,
                        help='probability that a random state is reported')
    parser.add_argument('--reward-noise-std', type=float, default=0.0,
                        help='standard deviation of Gaussian noise on the training reward')
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--eval-under-noise', action='store_true',
                        help='evaluate with the same noise applied; the default evaluates '
                             'on the clean task, which is usually the question being asked')
    parser.add_argument('--out', default='output_training/reference_run')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit("gymnasium is required: pip install -e '.[train]'") from exc

    if args.env.startswith('Gym-Gridworlds/'):
        # The gridworld ids register themselves on import of the external package.
        try:
            import gym_gridworlds  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                'that environment comes from the external Gym-Gridworlds package: '
                'pip install Gym-Gridworlds') from exc

    config = Config(n=None if args.n.lower() == 'mc' else int(args.n),
                    gamma=args.gamma, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                    episodes=args.episodes, max_steps=args.max_steps, seed=args.seed)
    noise = Noise(args.action_noise, args.observation_noise, args.reward_noise_std)

    env = gym.make(args.env)
    try:
        agent, history = train(env, config, noise)
        summary = evaluate(env, agent, episodes=args.eval_episodes,
                           seed=args.seed + 100_000, max_steps=args.max_steps,
                           gamma=args.gamma,
                           noise=noise if args.eval_under_noise else Noise())
    finally:
        env.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'episodes.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    with (out / 'summary.json').open('w') as handle:
        json.dump({
            'environment': args.env,
            'config': vars(args),
            'evaluated_under_noise': bool(args.eval_under_noise),
            'evaluation': summary,
            'source': 'corrected_rl reference implementation, not the archived 50-seed study',
        }, handle, indent=2)

    print(f"mean return over {summary['episodes']} greedy episodes: "
          f"{summary['mean_return']:.3f} (sd {summary['std_return']:.3f})")
    print(f'written to {out}')
    return summary


if __name__ == '__main__':
    main()
