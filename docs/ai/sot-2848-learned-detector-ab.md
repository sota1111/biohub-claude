# SOT-2848 — Learned TemporalUNet3D detector vs classical champion (leak-free LOFO CV A/B)

**Cycle 8, child B** of the biohub-claude Kaggle rank-improvement ladder (parent SOT-2846), following
SOT-2847 which verified that an offline GPU/attached-weights learned kernel is submittable in this Code
competition and landed a **default-off** learned-detector receptacle
(`src/biohub_tracking/learned_detect.py`).

## Hypothesis

The ~0.62 (classical public baseline) → ~0.89 (learned frontier) leaderboard gap is a **learned-vs-classical
detector-quality gap**. A trained `TemporalUNet3D` per-voxel detection head — with the champion
distance-only linker held fixed — should beat the classical DoG+NMS champion (leak-free CV micro-adj
**0.6649**) beyond the noise band with no per-dataset regression.

## Method (leak-free, leave-one-family-out)

A learned detector can leak if the scored video is in its training set (the classical champion has no
learned parameters and cannot leak). So we train **4 LOFO folds** — for each held-out family, train
`temporal_unet3d` on the *other three* families' sparse `.geff` GT and score the held-out family:

- Supervision: sum-of-Gaussians heatmap at each sparse GT node (voxel coords), foreground-weighted MSE
  on the sigmoid heatmap; patch `32×128×128`; per-volume standardisation identical to inference.
- Config: 1500 steps, seed 1234, `base=16`, `fg_weight=50`, RTX 3080 Ti. Final train losses 0.0027–0.0054.
- Scoring: the single leak-free harness `biohub_tracking.eval.cv` — **detection-only swap**, champion
  distance-only bipartite linker **fixed**, so the only difference is classical DoG+NMS vs learned heatmap.
- Scripts: `experiments/sot2848/train_lofo.py`, `experiments/sot2848/run_ab.py`.

## Result — REJECT

Champion self-check reproduced **exactly** inside the same run (micro-adj 0.6649; per-family adj
44b6_0113de3b=0.8895 / 44b6_0b24845f=0.6817 / 6bba_05b6850b=0.5700 / 6bba_05db0fb1=0.7310).

| Threshold | micro-adj | 44b6_0113de3b | 44b6_0b24845f | 6bba_05b6850b | 6bba_05db0fb1 |
| --- | --- | --- | --- | --- | --- |
| champion (classical) | **0.6649** | 0.8895 | 0.6817 | 0.5700 | 0.7310 |
| learned **0.5** (committed) | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| learned 0.3 (diagnostic) | 0.0423 | 0.8272 | 0.1999 | 0.0441 | 0.0000 |
| learned 0.7 (diagnostic) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**Decision: REJECT** — micro-adj delta **−0.6649**, `no_regression=False`. Result JSON:
`experiments/sot2848/screen_learned_ab.json`.

## Why the learned detector fails here

The learned sigmoid heatmap has **no family-robust operating point**:

- At the committed **0.5** it under-fires (44b6_0113de3b → 0 detections; 6bba_05db0fb1 → 13 detections /
  0 nodes), and where it does fire the blobs do not reconstruct GT edges (44b6_0b24845f → 614 detections
  but `edge_tp=0`).
- At **0.3** one fold over-detects wildly (44b6_0113de3b → 14827 detections / 13110 nodes, adj 0.8272 —
  tolerated only by that family's node-count-penalty headroom) while the other three collapse. Even this
  best sweep point (micro 0.0423) is an order of magnitude below the champion and regresses 3/4 families.

**Root cause** (consistent with the prior ledger): the competition train split ships **sparse** tracking
GT — ~1 labeled cell/frame for the two 44b6 videos, ~9–12 for the two 6bba videos, not dense masks. A
heatmap regressor on point supervision cannot learn a **calibrated, family-invariant** detection
magnitude from only 3 training families in-container; a single family-agnostic threshold does not exist.
This is the same sparse-GT PU-contamination that sank SOT-2828 (learned scorer) and the "no single
global operating point" wall of SOT-2830.

## Disposition

- **No promotion.** Champion `detect-link-dog-v4-shorttrack` `champion/config.json` kept **byte-frozen**
  (sha256 `42064648…`; `eval.cv --check-champion` delta 0.0000; full pytest suite **169 pass**).
- The `TemporalUNet3D` port lands **default-off** (registry arch `temporal_unet3d`, nothing enables it) as
  the frontier-detector receptacle + reproducible evidence.
- **No Kaggle submission.**

## Implication for SOT-2849

A bare per-voxel heatmap detector trained on this sparse GT is **not** the lever. The frontier gain most
likely requires the **learned linking / gap-recovery** (transformer association) and/or dense
pseudo-labels, not more detector-threshold tuning.
