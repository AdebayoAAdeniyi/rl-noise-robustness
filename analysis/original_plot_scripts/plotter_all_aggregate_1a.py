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


noise_combos = [
    (0.0, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.25, 0.0), (0.0, 0.5, 0.0),
    (0.0, 0.0, 0.1), (0.0, 0.0, 0.05), (0.0, 0.0, 0.01), (0.1, 0.0, 0.0),
    (1.0, 0.0, 0.0),(10.0, 0.0, 0.0)
]

noise_combos = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0),
                (0.0, 0.1, 0.0), (0.0, 0.25, 0.0), (0.0, 0.5, 0.0),
                (0.0, 0.0, 0.01), (0.0, 0.0, 0.05), (0.0, 0.0, 0.1)
    ]

worlds = [
    "CliffWalk-4x12-v0", "CliffWalk-5x5-v0", 
           "Penalty-3x3-v0"
           ]

worlds = ["DangerMaze-6x6-v0"]



world_y_ranges = {
    "CliffWalk-4x12-v0": (-0.1, 1.0),
    "CliffWalk-5x5-v0": (-3.0, 1.0),
    "DangerMaze-6x6-v0": (-0.1, 1.0),
    "Penalty-3x3-v0": (-0.1, 1.0),
    "CartPole-v1": (65, 100),  # Adjust the range for each world
    "MountainCar-v0": (-110,-50),
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
n_vals = [0, 1, 2, 3, 10]
n_seeds = 50
alphas = [0.01]
results_dir = "output_plots"
os.makedirs(results_dir, exist_ok=True)

metrics_to_normalize = {"weights_stability", "theta_stability", "performance"}
metrics_to_normalize = {"performance"}
#metrics_to_normalize =  {}

import numpy as np
import os
import matplotlib.pyplot as plt

colorblind_friendly_colors = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", 
    "#D55E00", "#CC79A7", "#999999", "#000000", "#FF007F"
]

# Aggregation and bar plot code
# Aggregation and combined bar plot code
for world in worlds:
    base_plots_dir = os.path.join(results_dir, world)
    os.makedirs(base_plots_dir, exist_ok=True)

    for alpha in alphas:
        aggregated_metrics = {metric: [] for metric in metrics_keys.keys()}  # Collect data for all noise_combos

        for reward_noise_sd, rand_act_prob, obs_noise in noise_combos:
            for metric, name in metrics_keys.items():
                print(f"Processing: World={world}, Metric={metric}, Alpha={alpha}, NoiseCombo={(reward_noise_sd, rand_act_prob, obs_noise)}")
                
                # Load and aggregate data
                try:
                    data = np.load(f"{world}_{n_seeds}_{alpha}_{reward_noise_sd}_{rand_act_prob}_{obs_noise}_results.npz")
                except FileNotFoundError:
                    print(f"File not found: {world}_{n_seeds}_{alpha}_{reward_noise_sd}_{rand_act_prob}_{obs_noise}_results.npz")
                    continue
                
                if metric not in data:
                    print(f"Metric '{metric}' not found in the file.")
                    continue
                
                results_metric = data[metric]
                if results_metric.ndim == 3:  # Expected: (n_vals, n_seeds, steps)
                    avg_values = [np.mean(results_metric[idx, :, :], axis=0) for idx in range(len(n_vals))]
                    std_devs = [np.std(results_metric[idx, :, :], axis=0) for idx in range(len(n_vals))]
                    
                    # Compute overall averages and std deviations across steps
                    avg_values = [np.mean(avg) for avg in avg_values]
                    std_devs = [np.mean(std) for std in std_devs]
                    
                    # Append data for this noise combo
                    aggregated_metrics[metric].append((avg_values, std_devs))
                else:
                    print(f"Unexpected shape for {metric}: {results_metric.shape}")
                    continue

        # Plot combined bar chart for each metric
        for metric, all_noise_data in aggregated_metrics.items():
            if not all_noise_data:
                continue
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # X-axis positions for the bars
            x_positions = np.arange(len(noise_combos))  # One position per noise_combo
            bar_width = 0.8 / len(n_vals)  # Split bar width among n_vals
            
            # Plot each n_val's bars
            for idx, n in enumerate(n_vals):
                avg_values = [data[0][idx] for data in all_noise_data]  # Extract avg for n_val
                std_devs = [data[1][idx] for data in all_noise_data]  # Extract std dev for n_val
                label = 'MC' if n == 0 else f'n = {n}'                
                ax.bar(
                    x_positions + idx * bar_width,
                    avg_values,
                    yerr=std_devs,
                    width=bar_width,
                    label=label,
                    capsize=5,
                    color=colorblind_friendly_colors[idx % len(colorblind_friendly_colors)]  # Color selection
                )

            # Configure plot aesthetics
            ax.set_xlabel("Noise (Reward, Action, Observation)", fontsize=12)
            ax.set_ylabel(metrics_keys[metric], fontsize=12)
            ax.set_title(f"{world} - {metrics_keys[metric]} Combined Bar Plot", fontsize=14)
            ax.set_xticks(x_positions + bar_width * (len(n_vals) - 1) / 2)
            ax.set_xticklabels([f"{combo}" for combo in noise_combos], rotation=45, ha='right', fontsize=12)
            ax.tick_params(axis='y', labelsize=12) 
            ax.legend(title="n-values", fontsize=10)

            if metric == "results_exp_ret" and world in world_y_ranges:
                ax.set_ylim(world_y_ranges[world])  # Set custom y-axis range for each world for "Expected return"

            plt.tight_layout()

            # Save the combined bar chart with high resolution (300 dpi)
            plot_path = os.path.join(
                base_plots_dir, f"{world}_{metric}_combined_barplot_{n_seeds}_{alpha}.png"
            )
            plt.savefig(plot_path, dpi=300)  # High resolution
            plt.close(fig)


