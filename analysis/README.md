# Analysis and figure generation

Two sets of scripts, with different purposes.

`original_plot_scripts/` holds the aggregation, bar-chart, learning-curve and
heatmap code that produced the figures in our report.
[`MANIFEST.md`](original_plot_scripts/MANIFEST.md) says which script made what,
since the names do not. They read the multi-seed
`.npz` arrays the trainers wrote, with one array per metric shaped
(5 horizons, 50 seeds, 600000 steps). The arrays themselves are not in this
repository; set `RL_NOISE_RUN_DIR` to a directory holding your own.

`plot_saved_results.py` is a small path-driven plotter for the same array
schema: pick a metric, it averages across the seed axis and writes a figure.
It validates the array shape rather than failing later with a confusing index
error.

Two of the metrics these scripts can plot, `delta_error` and `reward`, are
affected by a logging defect in the trainers. The same applies to
`average_return` and `performance`. See [../docs/FINDINGS.md](../docs/FINDINGS.md)
before drawing conclusions from them.
