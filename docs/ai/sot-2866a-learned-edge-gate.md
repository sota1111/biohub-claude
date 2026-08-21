# SOT-2870 — Learned edge-cost gate-EXPANSION (motion + shape/intensity) FN-edge recovery

**Cycle 7, child of SOT-2866 (external-knowledge escalation, ladder step-6).**
Verdict: **REJECTED / non-promotable** (default-off infra retained, champion byte-frozen,
no Kaggle submission).

## Hypothesis

The custom metric (nodes matched @7µm; an edge is TP only when *both* endpoints match a
GT-edge-joined pair; **no node FP** for unmatched predictions; score = adjusted edge
Jaccard + 0.1·division Jaccard, micro-averaged with an over-prediction penalty) makes
**linking dominate division ~10:1**, and the gains come from **FN-edge recovery**. Two
prior results framed the axis:

- **SOT-2841** (learned edge *re-ranker*, `label = edge`) = **REJECTED**: the logistic
  over `[dist, app_cos, rival counts, ranks, margin]` was genuinely discriminative
  (held-out `p_edge` gap 0.42) but gave **zero CV gain** — a *fixed raw-distance*
  feasible set is saturated (near ≈ GT), so re-ordering it recovers nothing.
- **SOT-2864** (ARGUS motion-model linking) = **+0.0111, 4-family non-regression**: the
  gain came from `motion_gate_on_prediction=True` — *admitting* raw-far but
  motion-consistent successors the distance gate drops. **The headroom is in the
  feasibility GATE, not the re-rank.**

So this issue tested the union: a learned edge cost whose features are extended with the
**SOT-2864 motion residual** (predicted-position distance) and two Trackastra-style
shallow **shape/intensity ratios** (descriptor brightness / spread change — reused from
the existing patch descriptor, so **no new extraction, no train/infer skew**), used to
drive a **gate-EXPANSION admissibility**: a pair beyond `max_distance` in raw distance is
admitted iff it is motion-corrected in-range (`dist_pred ≤ max_distance`), within a
bounded raw ratio (`≤ 1.5·max_distance`), **and** the classifier scores it a real edge
(`p_edge ≥ admit_prob`). Never an unbounded long-range edge (metric-validity preserved).

## Method

- `EDGE_FEATURE_NAMES` extended (7 → 10) with `motion_resid`, `intensity_ratio`,
  `shape_ratio`; `edge_feature_planes(..., dist_pred=…)` threads the motion-predicted
  distance so the feature vector matches the gate-expansion decision (feasible set = raw
  `≤max` **∪** motion-pred `≤max`). `dist_pred=None` reproduces the SOT-2841 planes
  byte-for-byte.
- `LinkParams.edge_gate_expand` / `edge_gate_admit_prob` / `edge_gate_expand_ratio`
  (default-off) implement the admissibility in `_assign`. Off ⇒ champion byte-for-byte.
- Leak-free 4-family LOFO (`biohub_tracking.eval.cv`, SOT-2817 re-anchored full metric).
  Training pairs span the **motion-corrected union**, so the raw-far GT edges (the FN
  edges the expansion must admit) are in the trainable set. Detection + descriptors
  cached once; every variant re-links off the cache (single-variable same-seed A/B).
- Screen: `experiments/sot2866a/screen_learned_gate.py` →
  `experiments/sot2866a/screen_learned_gate.json`.

## Result (leak-free CV, champion micro-adj 0.6649, sanity reproduced byte-exact)

| variant | score | Δ vs champion | per-dataset non-reg | edge Δ (TP/FP/FN) |
| --- | --- | --- | --- | --- |
| champion (distance-only) | 0.6649 | +0.0000 | — (floor) | 0/0/0 |
| motion_rerank (SOT-2864, raw gate) | 0.6670 | +0.0021 | — | — |
| **motion_gate (SOT-2864, predicted gate)** | **0.6760** | **+0.0111** | **✅ 4/4** | **+16/−20/−16** |
| learned_gate@0.5 (this issue, best) | 0.6684 | +0.0035 | ❌ | +5/−7/−5 |
| learned_gate@0.6 | 0.6677 | +0.0028 | ❌ | +2/−8/−2 |
| learned_gate@0.7–0.9 | 0.6670 | +0.0021 | ❌ | 0/−8/0 |

- **LOFO held-out `p_edge` gap = 0.4928** — even *more* discriminative than SOT-2841's
  0.42. Yet no promotable variant.
- Best learned_gate (@0.5) **regresses the clean `44b6_0113de3b`** family
  (0.8895 → 0.8892), **failing the mandatory per-dataset non-regression gate**, and is
  **strictly worse than the simpler pure motion gate** (+0.0035 vs +0.0111).
- Only a handful of raw-far GT edges even exist to recover
  (`expanded_pos(raw>max)` = 1 / 0 / 8 / 10 across the four families), so the ceiling
  for this lever is small to begin with.

## Why it fails (mechanism)

The pure motion gate (`gate_on_prediction=True`) admits **all** motion-corrected pairs
(dTP +16). The learned expansion keeps the base gate on the raw distance and admits only
the high-`p_edge` subset — a **strict subset** of the motion gate's admits, so it
recovers **fewer** FN edges (dTP +5 at the loosest τ, shrinking to 0 as τ rises), while
the learned filter also drops a real edge on the clean family (the 44b6 regression).

This reconfirms SOT-2841's lesson — **discriminative power ≠ CV improvement** — now
extended from the *re-rank* to the *feasibility GATE*: a 0.49-gap learned `p_edge` is a
**worse admission criterion than raw motion-consistency**. Motion residual already
carries the whole admissible signal; layering a learned classifier on top only subtracts.

## Decision

- **Non-promotion.** Champion `detect-link-dog-v4-shorttrack` `config.json`
  sha256 `42064648…` **byte-frozen**; `edge_gate_expand` ships **default-off** (a strict
  superset — champion reproduced byte-for-byte, asserted in tests). No Kaggle submission.
- **SOT-2864 motion-model linking (+0.0111, 4/4 non-regression) remains the #1
  reserve / final-window LB-probe candidate** — this axis did not beat it.
- Ladder implication: the learned-linking family (SOT-2841 re-rank, SOT-2870 gate) is
  exhausted for the FN-edge lever on the classical champion detections — the learned
  score adds nothing over raw motion geometry. Remaining escalation is a genuinely
  *global* windowed association (SOT-2871) or spending the endgame reserve to measure the
  missing CV→LB transfer coefficient for the SOT-2864 motion gate.
