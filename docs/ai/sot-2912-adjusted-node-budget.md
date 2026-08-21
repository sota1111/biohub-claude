# SOT-2912 — Adjusted-Jaccard node-budget operating-point calibration (cycle 3)

**Axis.** The official royerlab metric is the **adjusted** edge Jaccard — the raw
Jaccard scaled by an explicit node-budget penalty
`J_adj = max(0, J·(1 − 0.1·(N_pred − N_true)/N_true))` (metrics.md, α=0.1). Earlier
operating-point work (SOT-2789) swept the **raw** metric and was declared exhausted
*there*. This axis instead calibrates the operating point against the **adjusted
objective** directly: using the geff `estimated_number_of_nodes` metadata as the
per-family true-node budget `N_true`, sweep the operating point that maximises
`micro_adj` rather than `micro_raw`.

New eval helper `biohub_tracking.eval.node_budget` (pure, unit-tested): the
node-budget penalty algebra (`node_budget_penalty`, `penalty_free_pred_nodes`), a
GT-density cross-check (`gt_node_count`), and the adjusted-objective operating-point
selector/verdict (`select_adjusted_operating_point`). Two levers swept:

* **min_track_length** (primary) — the champion's post-link short-track prune, the
  node-budget lever on the *linking* side. Detection is byte-identical to the
  champion, so each family is detected once and re-linked per value.
* **mad_k** (secondary) — the adaptive detection threshold `median + mad_k·1.4826·MAD`;
  raising it detects fewer nodes (detection-side node budget). Re-detect per value.

Champion `champion/config.json` is **not** touched (byte-frozen, sha
`42064648…e01bdd`, verified in-run). No Kaggle submission; no candidate artifact
(rejected).

**Distinct from SOT-2789 (not a retry).** SOT-2789 exhausted the operating-point
sweep on the **raw** metric. This optimises the **adjusted** objective / node-budget
penalty term — a genuinely different selection surface, confirmed empirically below
(`raw_adj_divergent = True`).

## Result — REJECTED (same-seed A/B, family-mix wall)

`experiments/sot2912/ab_node_budget.py` → `ab_node_budget.json`. SOT-2903 re-anchored
4-family leak-free CV, same seed / same metric. primary=`micro_adj`, guardrail=`micro_raw`.

Champion (mtl=4): micro_adj **0.6649**, micro_raw 0.6840, total pred nodes 145 051 vs
`N_true` budget 134 712 (**7.7% over**).

| operating point | micro_adj | Δadj | micro_raw | total pred | adj 4/4 no-reg | raw 4/4 no-reg |
|:--|--:|--:|--:|--:|:--:|:--:|
| min_track_length=1 | 0.6232 | −0.0417 | 0.6531 | 161 556 | False | False |
| min_track_length=2 | 0.6430 | −0.0219 | 0.6694 | 156 001 | False | False |
| min_track_length=3 | 0.6602 | −0.0047 | 0.6833 | 150 487 | False | False |
| **min_track_length=4 (champion)** | **0.6649** | — | 0.6840 | 145 051 | **True** | **True** |
| min_track_length=5 | 0.6691 | +0.0042 | 0.6849 | 139 915 | False | False |
| min_track_length=6 | 0.6720 | +0.0071 | 0.6850 | 134 790 | False | False |
| min_track_length=7 | 0.6801 | +0.0152 | 0.6892 | 128 958 | False | False |
| min_track_length=8 | 0.6808 | +0.0159 | 0.6857 | 123 218 | False | False |
| mad_k=2.5 | 0.6652 | +0.0003 | **0.6935** | 156 706 | False | True |
| mad_k=3.5 | 0.6457 | −0.0192 | 0.6592 | 135 014 | False | False |

**Root cause = the same family-mix / sparse-GT wall the detection axes hit.**
The champion mtl=4 is the **unique** point that passes 4/4 per-dataset adjusted
non-regression. Trimming harder (mtl≥5) monotonically raises the *global* micro_adj
(+0.0159 at mtl=8) and drives the pred node count through the budget (mtl=6 ≈ budget,
mtl≥7 under budget), but **every** mtl≥5 fails 4/4 non-regression: the gain is
entirely the over-budget dense **6bba** lineage; the same prune removes real short
tracks in the sparse clean **44b6_0113de3b** family (the exact regression the champion
note recorded for mtl=5). `raw_adj_divergent = True` — the raw-optimal point is
mad_k=2.5 (micro_raw 0.6935, but over-budget so adj only 0.6652), the adjusted-optimal
is mtl=8 — so the adjusted objective *is* a different surface than the raw sweep, but
it still yields no robust, promotable operating point.

**Forensic (why the node-budget lever is weak).** The `gt_node_count` cross-check
shows the GT geff is **sparse** — only 52 / 51 / 861 / 1229 annotated nodes per family
— while the metric's `N_true` budget is the full-volume estimate 25 755 / 32 795 /
6 362 / 69 800. So the champion's penalty is modest (~7.7% over the coarse budget) and
satisfying it by pruning costs recall on the handful of real short tracks the sparse GT
actually scores. This is the same "sparse-GT over-count penalty ⇒ no single global
operating point" wall the detection axes (SOT-2774/2848/2863/2873/2884) documented.

## Verdict

REJECTED. Champion 0.6649 unchanged and byte-frozen; no candidate artifact. Ledger:
`docs/ai/experiment_ledger.jsonl` (cycle 3). The adjusted-objective calibrator lands as
a reusable, default-off eval helper for future cycles, with this negative result
recorded so the adjusted operating-point surface is not re-swept without new evidence
(a denser-GT re-anchor or a per-family operating point, not a single global one).
