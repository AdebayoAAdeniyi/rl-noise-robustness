# CartPole-v1

24 figures from our runs, main report environment.

Every file here is an original figure from the project. Filenames, byte
counts and SHA-256 digests are recorded in
[`../../ALL_FIGURE_PROVENANCE.json`](../../ALL_FIGURE_PROVENANCE.json), and
`tests/test_archive_layout.py` checks each digest.

## Read these with care

8 of the 24 figures here are built from a metric affected by the
logging defect described in [`../../../docs/FINDINGS.md`](../../../docs/FINDINGS.md).
They are kept rather than deleted, and listed so they are not read as sound:

- `CartPole-v1_average_return_combined_barplot_50_0.01.png`
- `CartPole-v1_delta_error_50_0.01_0.0_0.0_0.05_results.png`
- `CartPole-v1_delta_error_50_0.01_0.0_0.0_0.0_results.png`
- `CartPole-v1_delta_error_50_0.01_0.0_0.1_0.0_results.png`
- `CartPole-v1_delta_error_50_0.01_0.1_0.0_0.0_results.png`
- `CartPole-v1_delta_error_combined_barplot_50_0.01.png`
- `CartPole-v1_performance_combined_barplot_50_0.01.png`
- `CartPole-v1_reward_combined_barplot_50_0.01.png`

The remaining 16 figures cover expected return and the actor and critic
parameter variances. Those metrics do not pass through the faulty buffers and
are sound.

The code is in [`../../../experiments/`](../../../experiments/) and the
plotting code in [`../../../analysis/original_plot_scripts/`](../../../analysis/original_plot_scripts/).
The `.npz` arrays behind these plots are not in the repository.
