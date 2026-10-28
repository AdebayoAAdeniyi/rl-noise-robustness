import pandas as pd
import numpy as np
import altair as alt
from altair import theme
import os

# The run directory held the .npz metric arrays this script reads. It was a
# cluster scratch path on the machine these plots were produced on; set
# RL_NOISE_RUN_DIR to wherever your own .npz files are.
working_dir = os.environ.get("RL_NOISE_RUN_DIR", ".")
os.chdir(working_dir)


# @theme.register("param_font_size", enable=True)
# def custom_theme() -> theme.ThemeConfig:
#     sizes = 12, 14, 16, 18, 20
#     return {
        # "autosize": {"contains": "content", "resize": True},
        # "background": "#F3F2F1",
        # "config": {
        #     "axisX": {"labelFontSize": sizes[1], "titleFontSize": sizes[1]},
        #     "axisY": {"labelFontSize": sizes[1], "titleFontSize": sizes[1]},
        #     "font": "Bree Serif",
        #     "headerColumn": {"labelFontSize": sizes[1]},
        #     "headerFacet": {"labelFontSize": sizes[1]},
        #     "headerRow": {"labelFontSize": sizes[1]},
        #     "legend": {"labelFontSize": sizes[0], "titleFontSize": sizes[1]},
        #     "text": {"fontSize": sizes[0]},
        #     "title": {"fontSize": sizes[-1]},
        # },
#         "height": {"step": 28},
#         "width": 350,
#     }
alt.themes.enable('default')
def custom_theme():
    sizes =  18, 20, 24, 28, 30
    return alt.themes.ThemeConfig(
        {
            "autosize": {"contains": "content", "resize": True},
            "background": "#F3F2F1",
            "height": 400,
            "width": 700,
            "config": {
                "axisX": {"labelFontSize": sizes[1], "titleFontSize": sizes[2]},
                "axisY": {"labelFontSize": sizes[1], "titleFontSize": sizes[2]},
                "font": "Bree Serif",
                "headerColumn": {"labelFontSize": sizes[2]},
                "headerFacet": {"labelFontSize": sizes[2]},
                "headerRow": {"labelFontSize": sizes[2]},
                "legend": {"labelFontSize": sizes[2], "titleFontSize": sizes[2]},
                "text": {"fontSize": sizes[2]},
                "title": {"fontSize": sizes[4]},
            },
        }
    )



# @alt.theme.register('theme_name', enable=True)
# def custom_theme():
#     sizes = 12, 14, 16, 18, 20
#     return alt.theme.ThemeConfig(
#         {'height': 400, 'width': 700}
#     )

# ------------------------------------------------------------------------------
# Note that this script tries to use the Bree Serif font. To install it, check
# where matplotlib fonts are saved by running
#    from matplotlib.font_manager import findfont, FontProperties
#    print(findfont(FontProperties(family=["sans-serif"])))
#
# Then, download Bree Serif and install it there.
#
# Finally, delete matplotlib cache. To find it, run
#     matplotlib.get_cachedir()
# ------------------------------------------------------------------------------



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


worlds = [
    "CliffWalk-4x12-v0", "CliffWalk-5x5-v0",
           "Penalty-3x3-v0"
           ]

#worlds = ["CartPole-v1", "MountainCar-v0"]




n_seeds = 50
alpha = 0.001

# Define normalization functions
def normalize_data(data, method="z-score"):
    if method == "z-score":
        mean = np.mean(data, keepdims=True)
        std = np.std(data, keepdims=True)
        return (data - mean) / (std + 1e-8)
    elif method == "robust":
        median = np.median(data, keepdims=True)
        q75 = np.percentile(data, 75, keepdims=True)
        q25 = np.percentile(data, 25, keepdims=True)
        iqr = q75 - q25
        return (data - median) / (iqr + 1e-8)
    elif method == "min-max":
        min_val = np.min(data, keepdims=True)
        max_val = np.max(data, keepdims=True)
        return (data - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

normalization_method = "z-score"


# Configuration variables
#y_limits = [50, 100]

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
metrics_to_normalize = {"performance"}

# Output directory
results_dir = "output_plots_2"
os.makedirs(results_dir, exist_ok=True)


world_y_ranges = {
    "CliffWalk-4x12-v0": (-0.1, 1.0),
    "CliffWalk-5x5-v0": (-3.0, 1.0),
    "DangerMaze-6x6-v0": (-0.1, 1.0),
    "Penalty-3x3-v0": (-0.1, 1.0),
    "CartPole-v1": (60, 100),  # Adjust the range for each world
    "MountainCar-v0": (-110,-50),
}

range_labels = ["MC", "n=1", "n=2", "n=3", "n=10"]

# Loop through environments and metrics
for env in worlds:
    base_plots_dir = os.path.join(results_dir, env)
    os.makedirs(base_plots_dir, exist_ok=True)

    for metric, name in metrics_keys.items():
        # plots_dir = os.path.join(base_plots_dir, name)
        # os.makedirs(plots_dir, exist_ok=True)

        charts = []
        key_x = "Noise"
        key_col = "N-Step"
        key_y = name

        for i, sweep in enumerate([[0.0, 0.1, 0.5, 1.0], 
                                   [0.0, 0.1, 0.25, 0.5], 
                                   [0.0, 0.01, 0.05, 0.1]]):

            x, y, col = [], [], []
            title = ""

            for x_val in sweep:
                try:
                    # Determine file and title based on sweep type
                    if i == 0:
                        data = np.load(f"{env}_{n_seeds}_{alpha}_{x_val}_0.0_0.0_results.npz")
                        title = "Reward Noise"
                    elif i == 1:
                        data = np.load(f"{env}_{n_seeds}_{alpha}_0.0_{x_val}_0.0_results.npz")
                        title = "Action Noise"
                    elif i == 2:
                        data = np.load(f"{env}_{n_seeds}_{alpha}_0.0_0.0_{x_val}_results.npz")
                        title = "Observation Noise"
                except FileNotFoundError as e:
                    print(f"File not found: {e}")
                    continue

                results_metric = data[metric]

                


                # Normalize metrics if required
                normalize = metric in metrics_to_normalize
                if normalize:
                    normalization_method = normalization_method # Set the desired normalization method
                    results_metric = np.array([
                        normalize_data(results_metric[idx], method=normalization_method)
                        for idx in range(results_metric.shape[0])
                    ])

                # Process data for plotting
                for col_val in range(5):
                    tmp = results_metric[col_val, :, -1]
                    y += list(tmp)
                    x += [x_val] * len(tmp)
                    col += [col_val] * len(tmp)

            df = pd.DataFrame({key_x: x, key_y: y, key_col: [range_labels[val] for val in col]})

            # width = 600
            # height = 100
            # label_font_size = 18  # font of x/y ticks and labels (labels are in bold by default)
            # title_font_size = int(label_font_size * 1.8)

            # Determine y-axis limits based on metric and environment
            if metric == "results_exp_ret":
                y_limits = list(world_y_ranges.get(env, None))  
            else:
                y_limits =None
            print(y_limits)

            # Create Altair chart
            bars = alt.Chart(df).mark_bar(tooltip=True).encode(
                x=alt.X(f"{key_x}:N", axis=alt.Axis(labelAngle=0, title=title, labelFontSize=16, titleFontSize=18)),
                y=alt.Y(
                    f"mean({key_y}):Q",
                    scale=alt.Scale(domain=y_limits) if y_limits else alt.Undefined,
                    axis=alt.Axis(titleFontSize=18, labelFontSize=16)
                ),
                color=alt.Color(
                    f"{key_col}:N",
                    title="N-Step Values",
                    scale=alt.Scale(domain=range_labels),  # Explicitly define the order of labels
                    legend=alt.Legend(labelFontSize=16, titleFontSize=14)
                ),
                xOffset=alt.XOffset(f"{key_col}:N"),
            )

            error_bars = alt.Chart(df).mark_errorbar(extent="ci").encode(
                x=alt.X(f"{key_x}:N"),
                y=alt.Y(f"{key_y}:Q"),
                xOffset=alt.XOffset(f"{key_col}:N"),
            )

            combined = bars + error_bars
            charts.append(combined)

        chart = alt.vconcat(*charts).resolve_scale(x="independent")

        # Save the chart
        plot_path = os.path.join(base_plots_dir, f"bar_{env}_{metric}_{n_seeds}_{alpha}_results.png")
        chart.save(plot_path, dpi=300)

