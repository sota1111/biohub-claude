# SOT-2994 — Learned cross-attention edge linking (SimpleNodeTransformer port)

**Result: NON-PROMOTED (with evidence). Champion byte-frozen (config_sha256
`f2b1076…6522fc`, CV micro_adj 0.6760). Kaggle NOT submitted. Module ships
default-off.**

Cycle-7 (SOT-2992) direction 2 — the learned-linking half of the frontier pivot to
own-trained models. Runnable standalone (SOT-2993 learned detector is **not**
required; handcrafted geometric+intensity node features are the fallback).

## Axis

Replace the hand-tuned ARGUS motion-model LAP edge cost with a **learned
cross-attention edge scorer** ported from the official royerlab
`SimpleNodeTransformer`. Unlike the two REJECTED per-edge learned re-rankers, each
edge embedding is **contextualised by attention over the candidate set** before it is
scored:

* the per-edge feature vector is the proven `edge_feature_planes` (F=10) reused
  verbatim (no train/infer skew);
* row attention pools over each **source's competing successors**, column attention
  pools over each **destination's competing predecessors** (single-query masked
  softmax), injecting the candidate-set distribution into every edge score;
* a 2-layer head maps `[edge_embed, row_context, col_context] → p_edge`.

`p_edge` enters the champion cost as `dist + weight·(1 − p_edge)` — **re-rank only,
metric-valid**: the `≤ max_distance` motion feasibility gate is unchanged, so it
reorders the champion's feasible set and never admits an out-of-range edge.
`weight = 0` / no model / no descriptors ⇒ champion **byte-for-byte** (strict
default-off superset). Training is torch (float32, offline/dev-time); **inference is
pure numpy** (exec-compat, no torch/sklearn/pickle/`__file__`), pinned to the torch
forward by a parity test.

New-evidence vs the prior rejects (why the axis was reopened):

* **SOT-2841** — per-edge *logistic* re-ranker: held-out gap 0.42, **zero CV gain**
  (a fixed raw-distance feasible set is saturated; near ≈ GT).
* **SOT-2870** — the same logistic driving a motion+shape *gate expansion*: strictly
  worse than the plain SOT-2864 motion gate, regressed the clean 44b6 family.

The structural differentiator here is **set-context via cross-attention** (not an
independent per-pair classifier), and it learns the edge score rather than switching
a fixed operating point (≠ SOT-2922/2923/2931).

## Protocol

Leak-free leave-one-family-out (LOFO) same-seed A/B vs the frozen motion champion.
Detection frozen at champion params + descriptors, **cached once per family**
(linking-only ablation). Masked-sparse supervision (the official baseline's masked
loss): a trainable pair is a champion-feasible pair whose **source is GT-matched** —
positive iff it is the GT edge, else negative; unmatched-source pairs excluded (no
SOT-2828 node positive-unlabeled contamination). Each held-out family is linked by a
model trained ONLY on the other three families; the four held-out predictions
aggregate into the leak-free CV (`biohub_tracking.eval.cv`, primary micro_adj royerlab
adjusted edge Jaccard, guardrail micro_raw). Model coefficients are weight-independent,
so one fit per fold is swept over the re-rank strength. Training subsamples frame pairs
by stride-2 (a training-budget bound only; the **full CV is scored on every frame**);
>800 positive GT edges per fold remain. Screen:
`experiments/sot2994/ab_xattn_edge.py` → `experiments/sot2994/ab_xattn_edge.json`.

## Result (leak-free CV, champion micro_adj 0.6760 reproduced byte-exact)

Champion per-dataset adjusted edge Jaccard (the 4/4 non-regression floor):
44b6_0113de3b 0.9078 · 44b6_0b24845f 0.6938 · 6bba_05b6850b 0.5748 · 6bba_05db0fb1
0.7477.

| weight | micro_adj | Δ vs champion | 4/4 non-reg (adj) | edge Δ (TP/FP/FN) | which family moves |
|-------:|----------:|--------------:|:-----------------:|:------------------|:-------------------|
| 0.0 | 0.6760 | +0.0000 | ✅ (floor) | 0/0/0 | — (byte-identical) |
| 0.5 | 0.6767 | +0.0007 | ❌ | +1/−1/−1 | only 6bba_05b6850b (+0.0015) |
| 1.0 | 0.6760 | +0.0000 | ❌ | 0/0/0 | — |
| 2.0 | 0.6770 | +0.0010 | ❌ | +1/−2/−1 | only 6bba_05b6850b (+0.0021) |
| 4.0 | 0.6773 | +0.0013 | ❌ | +1/−3/−1 | only 6bba_05b6850b (+0.0021) |

**Held-out LOFO positive-vs-negative `p_edge` gap (discriminative power):**

| family | gap | n_pos / n_neg |
|--------|----:|---------------|
| 44b6_0113de3b | 0.6674 | 49 / 27 |
| 44b6_0b24845f | 0.5447 | 37 / 11 |
| 6bba_05b6850b | 0.4608 | 708 / 347 |
| 6bba_05db0fb1 | 0.5148 | 1032 / 654 |
| **mean** | **0.5469** | |

* `weight = 0` reproduces the champion **byte-for-byte** (dTP/dFP/dFN = 0/0/0).
* Mean held-out gap **0.5469** — the **most discriminative learned-edge model to
  date** (SOT-2841 0.42, SOT-2870 0.49): the cross-attention context genuinely sharpens
  the per-edge signal.
* **Yet no promotable variant.** Every positive weight moves only the single densest
  `6bba_05b6850b` family (+0.0015…+0.0021) while shaving at least one cleaner family
  below zero, so the mandatory **4/4 per-dataset non-regression gate fails** at every
  weight. Max micro_adj 0.6773 (+0.0013) is a family-mix artefact, not a genuine lift.

## Why it fails (mechanism)

This is the **third confirmation** of the SOT-2841/2870 lesson, now at the strongest
model class: **discrimination ≠ CV improvement on the classical detections.** The
champion's raw/motion feasibility gate already places the distance-optimal assignment
on the near ≈ GT agreement, so the feasible set has essentially no re-rank headroom. A
cross-attention scorer with the best held-out separation yet (0.55) still only flips a
handful of edges in the densest family — and doing so trips the same **family-mix /
density-mix wall** (SOT-2921): a re-rank that helps the crowded 6bba lineage regresses
a cleaner family, so no global re-rank strength clears 4/4. The learned-linking family
(logistic re-rank SOT-2841, learned gate SOT-2870, **cross-attention re-rank
SOT-2994**) is exhausted on the classical champion detections — the headroom is not in
re-ranking edges, it is in the **detection substrate** (SOT-2993 learned detector /
SOT-2992 direction-1) that would give the scorer genuinely richer node features and
change which edges are feasible in the first place.

## Decision

* **Non-promotion.** Champion `detect-link-dog-v4-shorttrack-motion-gain1`
  `config.json` sha256 `f2b1076…6522fc` **byte-frozen**; `xattn_edge_model` ships
  **default-off** (a strict superset — champion reproduced byte-for-byte, asserted in
  `tests/test_xattn_edge.py`). No Kaggle submission.
* The module, trainer, tests, and screen are kept as **default-off infrastructure**:
  once the SOT-2993 learned detector lands, this cross-attention scorer can consume its
  richer learned node features (the setting where the official `SimpleNodeTransformer`
  actually earns its keep) rather than the saturated classical handcrafted features.
* Ladder implication: further learned-*linking* re-rank on classical detections is a
  dead axis; the next learned-pivot compute belongs on the **detector** (SOT-2993) and
  on **joint detector+linker node features**, not on more edge re-rankers.
