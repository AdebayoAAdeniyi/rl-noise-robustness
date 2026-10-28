import gymnasium
import gym_gridworlds
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import os# The run directory held the .npz metric arrays this script reads. It was a
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


#world = "Penalty-3x3-v0"
world = "CliffWalk-4x12-v0"
#world = "CliffWalk-5x5-v0"
#world = "DangerMaze-6x6-v0"
reward_noise_sd = 0.0
rand_act_prob = 0.0
obs_noise = 0.0
# noise_combos = [(0.0, 0.0, 0.05), (0.0, 0.0, 0.1), (0.1, 0.0, 0.0),(0.0, 0.1, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.2)]
# reward_noise_sd, rand_act_prob, obs_noise = noise_combos[6]
n_seeds = 2
alpha = 0.001
MC_update = 0

# Load the saved results
data = np.load(f"{world}_{n_seeds}_{alpha}_{reward_noise_sd}_{rand_act_prob}_{obs_noise}_results.npz")

# Access saved variables
env_id = data["env_id"]
reward_noise_sd = data["reward_noise_sd"]
rand_act_prob = data["rand_act_prob"]
obs_noise = data["obs_noise"]
alpha_p = data["alpha_p"]
alpha_v = data["alpha_v"]
max_steps = data["max_steps"]
results_exp_ret = data["reward"]
#results_exp_ret = data["results_exp_ret"]
# print(len(results_exp_ret[0]))
# for arr in results_exp_ret[1]:
#     print(arr[-1])
# exit()
print("Type of theta_stability:", type(data["theta_stability"]))
print("Shape of theta_stability:", data["theta_stability"].shape if hasattr(data["theta_stability"], "shape") else "No shape")

# print("Type of results_exp_ret:", type(data["results_exp_ret"]))
# print("Shape of results_exp_ret:", data["results_exp_ret"].shape if hasattr(data["results_exp_ret"], "shape") else "No shape")


# Plot the loaded results
fig, axs = plt.subplots(1, 1)
axs.set_prop_cycle(color=["red", "green", "blue", "orange", "black", "purple"])
axs.set_xlabel("Steps")
axs.set_ylabel("Expected Return")

n_vals = [0, 1,2,3, 10]  # Update as needed
# n_seeds = results_exp_ret.shape[1]

index = 0
for n in n_vals:
    label = 'MC' if n == 0 else f'n = {n}'

    plot_args = dict(
        stepsize=1,
        smoothing_window=20,
        label=label,
    )
    error_shade_plot(
        axs,
        results_exp_ret[index],
        **plot_args,
    )
    axs.legend()
    index += 1

plt.title(f"{world}, seeds = {n_seeds}, reward noise std = {reward_noise_sd}, random action prob = {rand_act_prob}, obs noise = {obs_noise}")
plt.show()