# SOT-2995 — Official `evaluate.py` ported to the CV scorer + transfer-trust re-anchor

**Role:** cycle-2 foundational axis (role B) — give every learning experiment in
this cycle a *trustworthy* oracle by checking the local leak-free CV against the
**genuine** organiser scorer, not merely assuming they agree.

**Distinct from SOT-2929 (REJECTED).** SOT-2929 re-designed the holdout *grain*.
This axis does not touch the holdout — it repairs *oracle fidelity*: it runs the
real `royerlab/kaggle-cell-tracking-competition` scorer
(`tracking_cellmot.metrics` + `division_metrics`, which use `tracksdata` +
`polars`) on the same predictions and **measures** the divergence from the
clean-room reimplementation in `biohub_tracking.eval`.

## What was built

* **`src/biohub_tracking/eval/official.py`** — a bridge that runs the genuine
  official scorer on a `TrackingGraph` by building an in-memory `tracksdata`
  graph node-for-node / edge-for-edge, then calling the official
  `evaluate` / `per_sample_metrics` / `summarise` verbatim. Returns the same
  `EvaluationResult` / `FamilyResult` shapes as the clean-room path so the two
  are directly diff-able (`divergence_row`).
  * `tracksdata` / `tracking_cellmot` are **never** imported at module load —
    they are heavy, torch-adjacent, dev-only deps. Every entry point imports
    them lazily and raises `OfficialScorerUnavailable` if absent, so the light
    Kaggle-kernel path and CI are untouched (`official_available() == False`
    there). `eval/__init__` does not reference the module.
  * `graph_to_geff` — the light (`numpy`+`zarr`) `csv_to_geffs` counterpart:
    writes a `.geff` the official `scripts/evaluate.py` and
    `biohub_tracking.io.load_geff` both read (the csv side already lives in
    `io.load_submission_csv` / `write_submission_csv`, geff-read in
    `io.load_geff`).
* **`tests/test_official_metric.py`** — `graph_to_geff` round-trip (light,
  always runs) + the genuine scorer reproducing every frozen golden TP/FP/FN and
  agreeing count-for-count with the clean-room scorer (`importorskip`-guarded).
* **`experiments/sot2995/{predict,score_official,report}.py`** — the two-venv
  divergence harness (predict in the light repo `.venv`, score in a separate
  `tracksdata` venv) → `docs/ai/sot-2995-oracle-fidelity.json`.

## Result — the clean-room oracle is byte-faithful to the official scorer

**Divergence = 0** on every case scored, `max_abs_count_delta == 0`:

* **8 golden sandbox topologies** (perfect / missed / delayed / dummy-branch /
  spurious-linear / cross-component / grandchild-fallback / disconnected /
  `hack2` exploit) — official counts == frozen expected == clean-room.
* **8 real holdout predictions** — champion (motion-link, public **0.626**) and
  v1 (public **0.509**), each over the four leak-free CV families (dense 6bba +
  sparse 44b6, incl. division FN cases). Every family: `counts_match = True`.

| config | clean-room micro_adj | official micro_adj | Δ | public LB |
| --- | --- | --- | --- | --- |
| champion (motion-link) | 0.6760 | 0.6760 | 0.0 | 0.626 |
| v1 (global threshold) | 0.3598 | 0.3598 | 0.0 | 0.509 |

## Transfer-trust re-anchored on the official scorer

Re-scoring the two known submissions with the **official** metric and correlating
against the same-metric public anchors:

* official-scorer CV micro_adj: champion 0.6760 > v1 0.3598
* public LB: champion 0.626 > v1 0.509
* **Spearman ρ = 1.0**, `cv_top_matches_public_top = True` — the official-scorer
  CV orders the known submissions in agreement with the public leaderboard.

Because divergence is exactly zero, the clean-room order is identical, which
independently **confirms** the SOT-2903 transfer-trust finding (adjusted
micro-averaged edge Jaccard is the LB-faithful primary KPI) *under the genuine
organiser scorer* rather than under the reimplementation alone.

## Conclusion for the cycle

The self-made `micro_adj` reconstruction is **not** a generalization-gap culprit:
it reproduces the official `evaluate.py` to the count on both the organiser's own
reference topologies and real dense/sparse predictions. The leak-free CV remains
a faithful oracle (entity + temporal holdout unchanged; scoring only, never fit).
Learning experiments this cycle can trust the local CV; if CV↔LB still diverges,
the cause is representativeness/holdout-mix (SOT-2816/2817 guards) or genuine
non-transfer, **not** metric infidelity. No Kaggle submission was made.

Raw numbers: [`sot-2995-oracle-fidelity.json`](sot-2995-oracle-fidelity.json).
