# What each plotting script does

The project accumulated several variants of each plotter as the figure set
changed. They are kept because filenames in `results/figures/` trace back to
them, but it is not obvious from the names which is which.

| Script | Produces |
|---|---|
| `plotter_all_aggregate.py` | combined bar charts aggregating over noise levels |
| `plotter_all_aggregate_1a.py`, `_2.py` | the same, for later figure batches |
| `plotter_all_modified.py` | per-condition learning curves with confidence bands |
| `plotter_all_modified_1a.py`, `_2.py` | the same, for later figure batches |
| `plotter_new.py` | single-metric curves for one environment |
| `altair_concat.py`, `altair_cocat.py` | the normalised heatmap, Figure 1 of the report |
| `altair_cocat_1a.py`, `_2.py` | heatmap variants for later batches |
| `data_extractor.py` | pulls one metric out of a saved `.npz` into a table |

The `_1a` and `_2` suffixes mark successive passes over the figure set rather
than different methods. Where two of them differ, it is in which environments
and noise combinations are selected, not in how a quantity is computed.

All of them read `.npz` arrays from `RL_NOISE_RUN_DIR`. Those arrays are not
in this repository.
