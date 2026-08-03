# SOT-2369 — Short-track pruning (`detect-link-dog-v4-shorttrack`)

**Cycle 5.** Reference: the public Kaggle notebook
[*BioHub Clean Public-Frontier Lineage Tracker*](https://www.kaggle.com/code/prvsiyan/biohub-clean-public-frontier-lineage-tracker)
(claimed public score **0.913**).

## What was portable, and what was not

The reference notebook's 0.913 comes from a **GPU deep-learning pipeline**: a
pretrained dual-seed `TemporalUNet3D` detection/association field, a
`DeepCenterUNet3D` gap-repair model, a node transformer, and a constrained ILP
lineage solver (`tracksdata` + `pyscipopt` + `ilpy`). It hard-requires a CUDA T4+
GPU (`raise RuntimeError` otherwise), ~45 pip packages, and an external pretrained
model dataset. **None of that runs under this repo's submission constraint** —
`numpy`/`scipy`/`zarr` only, CPU, no internet, no pretrained weights, self-contained
Code-competition kernel. The notebook's own A/B/C dual-seed blend + retention guard
is worth only ~+0.0015 locally; the score lives in the pretrained models.

Two *classical* ideas from the notebook do transfer to our CPU pipeline. Both were
screened offline on the SOT-2305 4-dataset LB holdout (`44b6` ×2 + `6bba` ×2),
detector frozen at the DoG-v3 adaptive champion (`mad_k=3.0`):

1. **Motion-aware linking** — damped constant-velocity prediction
   `q̂ = q + gain·(q − q_prev)` in the assignment cost (the notebook's motion-aware
   reassignment). **REJECTED**: it helps the sparse `44b6` families but slightly
   hurts the dense `6bba` families that dominate the edge-count-weighted micro-avg,
   so the holdout micro-adj does not improve (0.6232 → ≤0.6218 across gains/gates).
   See `experiments/sot2369/screen_motion.json`.
2. **Short-track pruning** (`FILTER_SHORT_TRACKS`) — **PROMOTED** (below).

## The champion change

After linking, drop every weakly-connected track fragment with fewer than
`min_track_length = 4` nodes (`biohub_tracking.link._prune_short_tracks`). A real
cell persists across many frames, so a detection that never links into a ≥4-node
track is almost always noise.

This helps in **two** independent ways on the sparse light-sheet GT:

- **Node-count relief.** The adjusted edge Jaccard carries a penalty
  `J_adj = J·(1 − 0.1·(N_pred − N_true)/N_true)`. The champion over-predicts
  (e.g. `6bba_05b6850b`: 13 376 predicted vs ~6 362 true), so dropping spurious
  fragments raises the penalty factor toward 1.
- **Better matching.** The metric does per-timepoint optimal ≤7 µm node matching.
  A spurious fragment node can *win* a GT match and push the true track's detection
  to FN (and itself to FP). Removing fragments frees those GT nodes, so matched-edge
  **TP rises** while FP/FN fall — not just a node-count artifact.

## Result (SOT-2305 4-dataset LB holdout)

`min_track_length` sweep, holdout micro-adjusted edge Jaccard:

| mtl | 44b6 | 6bba | micro | note |
|----:|-----:|-----:|------:|------|
| 1 (incumbent) | 0.7716 | 0.6168 | **0.6232** | DoG-v3 adaptive champion |
| 2 | 0.7756 | 0.6372 | 0.6430 | |
| 3 | 0.7797 | 0.6549 | 0.6602 | |
| **4 (champion)** | **0.7836** | **0.6596** | **0.6649** | all datasets improve, no regression |
| 5 | 0.7689 | 0.6647 | 0.6692 | micro↑ but `44b6_0113de3b` TP 47→45 (real tracks pruned) |

Per-dataset adjusted edge Jaccard at mtl=4 — **every dataset improves**:

| dataset | incumbent | mtl=4 | edge TP/FP/FN (incumbent → mtl=4) |
|---|---:|---:|---|
| `44b6_0113de3b` | 0.8814 | 0.8895 | 47/2/3 → 47/2/3 (node-count relief) |
| `44b6_0b24845f` | 0.6658 | 0.6817 | 37/5/12 → 37/5/12 |
| `6bba_05b6850b` | 0.5025 | 0.5700 | 619/251/226 → 651/215/194 (**matching↑**) |
| `6bba_05db0fb1` | 0.7096 | 0.7310 | 963/166/220 → 973/148/210 |

`mtl=5` scored a hair higher on micro (0.6692) but **regressed the clean
`44b6_0113de3b` family** (TP 47→45, adj 0.8895→0.8539) by pruning real short tracks
— a generalization red flag. `mtl=4` is chosen as the largest cutoff with
**no per-dataset regression**.

## Gate

- Screen: `experiments/sot2369/screen_shorttrack.py` → `screen_shorttrack.json`
  (min_track_length sweep off cached detections).
- Confirm: `experiments/sot2369/confirm_shorttrack.py` → `confirm_shorttrack.json`
  (independent re-score through the real `champion_params() → run_pipeline`, which
  also validated that `link.min_track_length` reaches `LinkParams` — it caught a
  plumbing bug where `champion_params` was dropping the key).
- `min_track_length=1` reproduces the previous champion **byte-for-byte** (default).
- 55 pytest + exec-compat gate pass; `EMBEDDED_CHAMPION_CONFIG == champion/config.json`.
