# SOT-2828 — Portable GT-learned detection scorer (REJECTED, default-off knob kept)

**Cycle 3 (parent SOT-2827). Champion `detect-link-dog-v4-shorttrack` (micro-adj
0.6649) MAINTAINED byte-for-byte. No Kaggle submission.**

## Axis

The official royerlab baseline wins detection by a **GT-learned appearance
detector** (TemporalUNet3D). Every *unsupervised, global* operating-point lever on
top of the champion's `mad_k=3.0` NMS threshold was rejected across cycles 2-5
(multiscale DoG 2774, quantile-norm 2776, watershed 2775, density-gated split 2792,
local-MAD surface 2791, Hessian-blobness filter 2793) — a single global image
criterion cannot separate true nuclei from over-detections here. This is the untried
**supervised** axis ported to portable classical ML: a pure-numpy logistic scorer
learned from the sparse GT that re-ranks / selects each NMS candidate on a joint
hand-crafted feature vector, to cut FP without a new global threshold.

## What was built (kept, default-off; the durable mechanism)

* `src/biohub_tracking/detect_scorer.py` — feature extraction
  (`extract_candidate_features`: DoG response robust z-score + SOT-2829 appearance
  patch stats + Hessian blobness eigen-ratios + local neighbour density = 14
  features), the embedded pure-numpy `LearnedScorer` (`standardize → dot → sigmoid`,
  no sklearn/pickle/file at inference — Kaggle-kernel safe), sparse-GT
  positive-unlabeled labeling (`label_candidates`), and a numpy gradient-descent
  logistic trainer (`fit_scorer`; uses sklearn only if importable at train time).
* `src/biohub_tracking/detect.py` — new default-off `DetectParams.detect_scorer`
  post-NMS keep-mask (applied after `blobness_filter`, before `density_gated_split`);
  factored `_compute_response` shared with the champion path.
* `src/biohub_tracking/champion.py` — config plumbing (`detect.detect_scorer` dict
  passes through; absent ⇒ champion byte-identical).
* `scripts/train_detect_scorer.py` — train on all four families → embeddable JSON.
* `experiments/sot2828/screen_detect_scorer.py` — the leave-one-family-out A/B.
* `tests/test_detect_scorer.py` (10 tests) — feature shape/determinism,
  byte-invariance, threshold behaviour, fit separates synthetic signal, config
  round-trip, exec-compat (config-embedded scorer runs under `exec()`/no-`__file__`),
  sparse-GT labeling.

`champion/config.json` is unchanged, so the champion detector is byte-for-byte the
same (`eval.cv --check-champion` = 0.6649 exactly; 144 pytest; exec-compat gate OK).
As with SOT-2369 `velocity_gain`, SOT-2762 division and SOT-2793 blobness, the
rejected mechanism is retained as a documented default-off knob, not deleted.

## Evaluation — leave-one-family-out (leak-free)

A learned model scored on the same family it was fit on would leak, so each of the
four holdout families is scored with a logistic trained **only on the other three**.
Held-out per-family predictions are aggregated through the SOT-2817 re-anchored
full-metric CV. A probability keep-threshold sweep is a same-seed A/B (detection +
features cached once, only the keep-mask changes). `threshold=0.0` keeps every
candidate.

Sparse-GT class balance (positives = candidates matching the annotated **tracked
lineage** within 7µm): 44b6_0113de3b 54/32147 (0.2%), 44b6_0b24845f 51/42080 (0.1%),
6bba_05b6850b 1251/13376 (9.4%), 6bba_05db0fb1 similar. The GT annotates only the
tracked lineage, so "positive" ≈ one specific cell, and almost every *real*
un-annotated cell is a PU-contaminated "negative".

## Result — DECISIVELY REJECTED

| threshold | score | 44b6_0113 | 44b6_0b24 | 6bba_05b6 | 6bba_05db | ΣΔTP | ΣΔFP | ΣΔFN |
|-----------|-------|-----------|-----------|-----------|-----------|------|------|------|
| 0.0 (all) | **0.6649** | 0.8895 | 0.6817 | 0.5700 | 0.7310 | 0 | 0 | 0 |
| 0.3 | 0.3034 | 0.8895 | 0.6817 | 0.1672 | 0.3651 | −1048 | −220 | +1048 |
| 0.4 | 0.0968 | 0.8895 | 0.6818 | 0.0169 | 0.0933 | −1507 | −327 | +1507 |
| 0.5 | 0.0399 | 0.8896 | 0.6821 | 0.0026 | 0.0000 | −1622 | −361 | +1622 |
| 0.8 | 0.0050 | 0.0419 | 0.1731 | 0.0000 | 0.0000 | −1698 | −370 | +1698 |

* **Byte-invariance confirmed**: `threshold=0.0` reproduces the champion exactly
  (0.6649, ΔTP/ΔFP/ΔFN = 0). The scorer plumbing is metric-neutral when off.
* **No promotable variant at any threshold**; every one regresses, monotone with
  aggressiveness.

### Root cause — the learned scorer has ~zero cross-family discriminative power

The LOFO **held-out positive-minus-negative mean-probability gap is ≈ 0 or
negative** on every family: 44b6_0113de3b −0.030, 44b6_0b24845f −0.011,
6bba_05b6850b −0.105, 6bba_05db0fb1 +0.035. Positives (the tracked lineage) do **not**
score higher than the mass of unlabeled candidates on an unseen family — often lower.
So a probability threshold removes signal and noise indiscriminately:

* At `t=0.3` the dense families lose overwhelmingly **true** edges, not false ones —
  6bba_05b6850b ΔTP **−496** vs ΔFP −126, 6bba_05db0fb1 ΔTP **−552** vs ΔFP −94. The
  node-count relief (pred_nodes −5045 / −50742) cannot pay for the destroyed matched
  edges; adjusted Jaccard collapses 0.5700→0.1672 and 0.7310→0.3651.
* The 44b6 families barely filter at moderate thresholds (fold intercepts push their
  probabilities up), and at high thresholds they too collapse (0.8895→0.0419 at
  t=0.8). A tiny, non-robust +0.0005…+0.0026 44b6 FP-trim at t=0.5-0.6 exists but is
  within noise and never offsets the 6bba destruction — so **no single global
  threshold is per-dataset non-regressing**, exactly the family-mix gate that killed
  every prior detection lever.

**Interpretation.** The sparse GT (one annotated lineage) gives a positive-unlabeled
label set whose "positive" is a specific cell rather than "a real cell", and the
hand-crafted candidate features do not separate the annotated lineage from all other
detections in a way that **transfers to an unseen family**. This extends the
exhausted single-global-lever finding from the *unsupervised threshold* direction
(2774/2775/2776/2791/2792/2793) to the *supervised classifier* direction: on this
data and metric, candidate **selection** — learned or not — cannot beat the champion.
The next axis must leave detection selection (e.g. learned *linking* / appearance
matching where the label is an edge not a lone node, or dense pseudo-labels, or an
oracle/metric rebuild), not another candidate-scoring knob.

Artifact: `experiments/sot2828/screen_detect_scorer.json` (+ `.log`).
