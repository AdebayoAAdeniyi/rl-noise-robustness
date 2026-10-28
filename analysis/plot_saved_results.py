"""Plot a stored experiment metric from the original .npz output schema.

The provided archive contains PNG figures, not the numerical NPZ source arrays.
Do not use this utility to infer numerical results from image pixels.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = {"results_exp_ret", "reward", "average_return", "weights_stability", "theta_stability", "performance", "delta_error", "episode"}
LABELS = ["MC", "n=1", "n=2", "n=3", "n=10"]

def plot_metric(input_path, metric, output, stride=1000):
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    if stride <= 0:
        raise ValueError("stride must be positive")
    with np.load(input_path, allow_pickle=False) as data:
        if metric not in data:
            raise KeyError(f"{metric} not in {input_path}")
        x = data[metric]
    if x.ndim != 3 or x.shape[0] != 5:
        raise ValueError(f"Expected (five algorithms, seeds, steps); got {x.shape}")
    # Average runs over the seed axis. Do not confuse seed variation with CI.
    mean = np.mean(x, axis=1)
    steps = np.arange(0, x.shape[2], stride)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, label in enumerate(LABELS):
        ax.plot(steps, mean[i, ::stride], label=label)
    ax.set(xlabel="Environment steps", ylabel=metric, title=f"{metric}: mean across {x.shape[1]} seeds")
    ax.legend();fig.tight_layout()
    output = Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=170);plt.close(fig)

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("npz",type=Path);p.add_argument("--metric",choices=sorted(METRICS),default="results_exp_ret");p.add_argument("--output",type=Path,required=True);p.add_argument("--stride",type=int,default=1000)
    a=p.parse_args();plot_metric(a.npz,a.metric,a.output,a.stride)
