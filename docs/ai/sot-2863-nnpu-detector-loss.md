# SOT-2863 — PU-aware detector loss (nnPU / Cellsparse) vs classical champion

Cycle-9 child A of SOT-2862. External-knowledge port (ladder step-6). Attacks the
root cause SOT-2848 identified for its rejected learned detector: the **sparse
tracking GT** makes a naive foreground-weighted MSE treat every unlabeled cell
voxel as background — **Positive-Unlabeled (PU) contamination** — so the learned
heatmap has no family-robust operating point (micro-adj 0.0 at thr 0.5).

## Method (same-seed, single variable = the training loss)

`experiments/sot2862a/train_lofo.py` reuses SOT-2848's LOFO harness **byte-identically**
(temporal_unet3d, PATCH 32×128×128, 1500 steps, batch 2, lr 1e-3, base 16, seed 1234,
RTX 3080 Ti) and swaps **only** the loss:

- `naive` — SOT-2848 foreground-weighted MSE (reuses `experiments/sot2848/weights`).
- `nnpu` — non-negative PU risk `R̂ = π·R̂_P(+1) + max(0, R̂_U(−1) − π·R̂_P(−1))`
  (Kiryo et al. arXiv:1703.00593; PU cell detection arXiv:2106.15918). Blob-core
  voxels = labeled Positives, everything else = Unlabeled; the non-negative clamp
  stops the model overfitting unlabeled true-positives as negatives. `π` estimated
  from the **estimated-true** node density (`geff_estimated_num_nodes`), 2.8e-3–5.6e-3.
- `cellsparse` — Cellsparse ignore-weighting (biorxiv 2023.06.13.544786): naive MSE
  but unlabeled voxel loss down-weighted to 0.05.

`run_ab.py` scores champion + all three arms through the SAME leak-free harness
(`biohub_tracking.eval.cv`), detection-only swap, champion distance-only linker fixed.

## Result — REJECT (champion not beaten), root cause CONFIRMED

| arm | micro-adj (thr 0.5) | 44b6_0113de3b | 44b6_0b24845f | 6bba_05b6850b | 6bba_05db0fb1 |
|---|---|---|---|---|---|
| **champion (classical)** | **0.6649** | 0.8895 | 0.6817 | 0.5700 | 0.7310 |
| naive (SOT-2848) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| nnpu | 0.1173 | 0.7346 | 0.2520 | 0.1703 | 0.0466 |
| cellsparse | 0.2753 | 0.7627 | 0.0000 | 0.6036 | 0.0111 |

- **`nnpu_vs_naive` micro-delta = +0.1173**, and nnPU fires non-zero on **all four
  families** (vs naive's all-zero). The non-negative clamp **cured** the degenerate
  under-firing SOT-2848 diagnosed → **PU-contamination is confirmed as the naive
  detector's failure root cause** (this is the new evidence justifying the axis).
- But **no learned arm reaches champion 0.6649** at any single family-invariant
  operating point. Cellsparse (best, 0.2753) collapses on 44b6_0b24845f=0.0000; the
  threshold sweep shows the same "no single operating point on sparse GT" wall
  (SOT-2830/2848/2849): nnpu thr0.3 44b6_0113de3b=0.8155 but 6bba_05db0fb1=0.0840;
  thr0.7 collapses both.

## Decision

- **REJECT** all learned arms; champion `detect-link-dog-v4-shorttrack`
  `config.json` sha256 `42064648…` **byte-unchanged**; learned detector stays
  **default-off** (no shipped-module change — the loss lives only in the offline
  experiment harness).
- **No Kaggle submission** (parent resume run's responsibility).
- Gates: **178 pytest pass**, compileall OK, `exec_compat_gate` OK.

## Implication

The detector-quality ladder on this sparse GT is exhausted for a single-operating-point
per-voxel heatmap **regardless of loss** — naive is degenerate, nnPU/Cellsparse recover
a working detector on 3/4 families but cannot calibrate a family-invariant magnitude
from 3 sparse-GT training families. Frontier gain now needs either dense/pseudo-label
supervision or the learned **linking** axis. SOT-2864 motion-model linking (+0.0111 CV,
all-family non-regressing) remains the standing #1 reserve / final-window LB-probe.
