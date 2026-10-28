import gymnasium
import gym_gridworlds
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import os

# The run directory held the .npz metric arrays this script reads. It was a
# cluster scratch path on the machine these plots were produced on; set
# RL_NOISE_RUN_DIR to wherever your own .npz files are.
working_dir = os.environ.get("RL_NOISE_RUN_DIR", ".")
os.chdir(working_dir)


# https://stackoverflow.com/a/63458548/754136
def smooth(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    re[0] = arr[0]
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span :])
    return re

def error_shade_plot(ax, data, stepsize, smoothing_window=1, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    if smoothing_window > 1:
        y = smooth(y, smoothing_window)
    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        error = smooth(error, smoothing_window)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())


metrics_keys = {
    "reward": "Reward",
    "results_exp_ret": "Expected return",
    "average_return": "Average return",
    "weights_stability": "Weights variance",
    "theta_stability": "Theta variance",
    "performance": "Performance variance",
    "episode": "Episodes",
}

metrics_keys = {
    "reward": "Reward",
    "results_exp_ret": "Expected return",
    "average_return": "Average return",
    "weights_stability": "Weights variance",
    "theta_stability": "Theta variance",
    "performance": "Performance variance",
    "delta_error": "Delta error",
    "episode": "Episodes",
}


results_dir = "output_plots"
os.makedirs(results_dir, exist_ok=True)

# Define normalization function
def normalize_data(data, method="z-score"):
    if method == "z-score":
        mean = np.mean(data, keepdims=True)
        std = np.std(data, keepdims=True)
        return (data - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero
    elif method == "robust":
        median = np.median(data, keepdims=True)
        q75 = np.percentile(data, 75, keepdims=True)
        q25 = np.percentile(data, 25, keepdims=True)
        iqr = q75 - q25
        return (data - median) / (iqr + 1e-8)  # Add small epsilon to avoid division by zero
    elif method == "min-max":
        min_val = np.min(data, keepdims=True)
        max_val = np.max(data, keepdims=True)
        return (data - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

# Define metrics to normalize



worlds = ["CartPole-v1", "MountainCar-v0"]
worlds = ["MountainCar-v0"]

world_y_ranges = {
    "CliffWalk-4x12-v0": (-0.1, 1.0),
    "CliffWalk-5x5-v0": (-3.0, 1.0),
    "DangerMaze-6x6-v0": (-0.1, 1.0),
    "Penalty-3x3-v0": (-0.1, 1.0),
    
    "MountainCar-v0": (-110,-50),
}


noise_combos = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), 
(0.0, 0.0, 0.1),
                (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.05), 
                (0.75, 0.0, 0.0), (0.0, 0.75, 0.0), (0.0, 0.0, 0.01), 
                (0.0, 0.95, 0.0), (1.0, 0.0, 0.0), (0.0, 0.25, 0.0), (10.0, 0.0, 0.0),]


noise_combos = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), 
(0.0, 0.0, 0.1),
                (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.05), 
                ]

noise_combos = [(1.0, 0.0, 0.0), (0.0, 0.25, 0.0)]

#noise_combos = [
#    (1.0, 0.0, 0.0), (0.0, 0.25, 0.0)]

metrics_keys = {
    "reward": "Reward",
    "results_exp_ret": "Expected return",
    "average_return": "Average return",
    "weights_stability": "Weights variance",
    "theta_stability": "Theta variance",
    "performance": "Performance variance",
    "delta_error": "Delta error",
    "episode": "Episodes",
}
n_vals = [0, 1, 2, 3, 10]
n_seeds = 5
alphas = [0.01]
results_dir = "output_plots"
os.makedirs(results_dir, exist_ok=True)

metrics_to_normalize = {"weights_stability", "theta_stability", "performance"}
metrics_to_normalize = {"performance"}
#metrics_to_normalize =  {}

#"z-score", "robust", "min-max"
normalization_method = "z-score"
# Plotting and normalization loop
import numpy as np
import os
import matplotlib.pyplot as plt

# Color Universal Design (CUD) palette for color blindness
colorblind_friendly_colors = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", 
    "#D55E00", "#CC79A7", "#999999", "#000000", "#FF007F"
]



# Aggregation and plotting code
for world in worlds:
    base_plots_dir = os.path.join(results_dir, world)
    os.makedirs(base_plots_dir, exist_ok=True)

    for alpha in alphas:
        for reward_noise_sd, rand_act_prob, obs_noise in noise_combos:
            for metric, name in metrics_keys.items():
                print(f"Processing: World={world}, Metric={metric}, Alpha={alpha}, NoiseCombo={(reward_noise_sd, rand_act_prob, obs_noise)}")
                
                # Directory for each metric
                plots_dir = os.path.join(base_plots_dir, name)
                os.makedirs(plots_dir, exist_ok=True)

                # Load data
                try:
                    data = np.load(f"{world}_{n_seeds}_{alpha}_{reward_noise_sd}_{rand_act_prob}_{obs_noise}_results.npz")
                except FileNotFoundError:
                    print(f"File not found: {world}_{n_seeds}_{alpha}_{reward_noise_sd}_{rand_act_prob}_{obs_noise}_results.npz")
                    continue
                
                results_metric = data[metric]

                # Apply normalization if the metric is in the list
                normalize = metric in metrics_to_normalize
                if normalize:
                    normalization_method = normalization_method  # Change to "robust" or "min-max" if preferred
                    results_metric = np.array([
                        normalize_data(results_metric[idx], method=normalization_method) 
                        for idx in range(results_metric.shape[0])
                    ])

                # Create a new figure for each plot
                fig, ax = plt.subplots(figsize=(6, 4))
                title_font = {'fontsize': 9, 'fontweight': 'bold', 'family': 'serif'}
                axis_font = {'fontsize': 12, 'family': 'sans-serif'}

                ax.set_xlabel("Steps", fontdict=axis_font)
                ax.set_ylabel(f"{name} {'(Normalized)' if normalize else ''}", fontdict=axis_font)

                # Use the colorblind-friendly palette for the plot
                ax.set_prop_cycle(color=colorblind_friendly_colors)

                # Plot results
                for idx, n in enumerate(n_vals):
                    label = 'MC' if n == 0 else f'n = {n}'
                    plot_args = dict(
                        stepsize=1,
                        smoothing_window=20,
                        label=label,
                    )
                    error_shade_plot(
                        ax,
                        results_metric[idx],
                        **plot_args,
                    )
                
                # Set the title and legend
                ax.legend()
                ax.set_title(f"{world}\n {name} {'(Normalized)' if normalize else ''}\n Reward Noise Std={reward_noise_sd}, Random Action Prob={rand_act_prob}, Obs Noise={obs_noise}", fontdict=title_font)
                
                # Set y-axis limits for the Expected Return metric (only for "results_exp_ret")
                if metric == "results_exp_ret" and world in world_y_ranges:
                    ax.set_ylim(world_y_ranges[world])  # Set custom y-axis range for each world for "Expected return"

                plt.tight_layout()

                # Save the plot with high resolution (300 dpi)
                plot_path = os.path.join(
                    plots_dir, f"{world}_{metric}_{n_seeds}_{alpha}_{reward_noise_sd}_{rand_act_prob}_{obs_noise}_results{'_normalized' if normalize else ''}.png"
                )
                plt.savefig(plot_path, dpi=300)  # Save with 300 dpi for high resolution
                plt.close(fig)

