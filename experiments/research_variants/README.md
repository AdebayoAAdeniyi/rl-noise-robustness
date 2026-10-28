# Variants and reruns

Working versions from the project: parallel and serial runners, the data
savers for each environment family, a combined CartPole and MountainCar
runner, exploratory MountainCar changes, and two later reruns.

They are kept because they are part of how the study was actually built, and
because several of them differ from the main trainers in ways that matter if
you compare their outputs: different seed counts, different noise grids,
different step budgets. None of them is the source of a figure in the report
unless its filename matches an entry in
[`../../results/ALL_FIGURE_PROVENANCE.json`](../../results/ALL_FIGURE_PROVENANCE.json).

They share the metric-logging defect described in
[`../../docs/FINDINGS.md`](../../docs/FINDINGS.md).
