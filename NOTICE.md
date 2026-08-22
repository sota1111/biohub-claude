# Attribution

The local evaluator in `src/biohub_tracking/eval/` is a clean-room
reimplementation of the official competition metric for
**Biohub — Cell Tracking During Development**
(<https://www.kaggle.com/competitions/biohub-cell-tracking-during-development>).

The metric definition and the algorithmic rules for the edge and division
Jaccard were derived from the competition organiser's public specification and
reference implementation:

- `royerlab/kaggle-cell-tracking-competition` — `metrics.md`,
  `src/tracking_cellmot/metrics.py`, `src/tracking_cellmot/division_metrics.py`
  (BSD-3-Clause, Copyright (c) 2026 Thibaut Goldsborough).

The eight hand-crafted graph cases and their frozen expected TP/FP/FN counts in
`tests/test_sandbox_golden.py` are ported from that project's
`tests/test_division_sandbox_examples.py` and are used here as a conformance
check that this reimplementation matches the official metric.

This repository does **not** vendor that project's code (which depends on
`torch` / `tracksdata`); the evaluator here is an independent, dependency-light
(numpy + scipy) implementation.

`src/biohub_tracking/eval/official.py` (SOT-2995) is a **dev-only bridge** that
runs the genuine organiser scorer (`tracking_cellmot.metrics` /
`division_metrics`, via `tracksdata` + `polars`) on this repo's predictions to
measure fidelity against the clean-room implementation. It imports those heavy
deps lazily, so it is never pulled into the light Kaggle-kernel path; it does not
vendor their source. The measured divergence is zero on all golden topologies and
real holdout predictions (`docs/ai/sot-2995-oracle-fidelity.md`).
