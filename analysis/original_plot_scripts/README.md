# The plotting code behind the report figures

These are the scripts that produced the figures in our report: metric
aggregation, per-condition learning curves, combined bar charts across noise
levels, and the normalised heatmap in Figure 1.

They are kept as they were written, with one change: the hard-coded run
directory has been replaced by the `RL_NOISE_RUN_DIR` environment variable, so
no machine-specific path is published. Point it at a directory holding the
`.npz` arrays a trainer wrote.

They are not portable entry points. Each one targets a specific result batch,
with environment names, noise grids and file naming fixed inline. Read the
bottom of a script before running it. For a general-purpose plot of one metric
across seeds, use [`../plot_saved_results.py`](../plot_saved_results.py).
