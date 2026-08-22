# SOT-2993 — Self-trained 3D U-Net detector with masked sparse loss

Parent SOT-2992 ([biohub-claude] Kaggle 順位向上サイクル 第2次). Path-A of the
learned-detector escalation: replace the classical DoG+NMS detection stage with a
self-trained `TemporalUNet3D`, trained with **masked sparse supervision**, and A/B
it against the classical champion on the single leak-free CV. Default-off; classical
champion byte-frozen and preserved as the hedge; **no Kaggle submission** (submission
is the parent-resume run's decision).

## Why this is new grounds (not a blind SOT-2828/2848/2863 retry)

The learned-detector axis has been evaluated three times and rejected each time; the
shared root cause is the **sparse-GT Positive-Unlabeled (PU) contamination**:

| Prior | Loss | Leak-free micro-adj | Failure mode |
| --- | --- | --- | --- |
| SOT-2828 | pure-numpy logistic scorer (re-rank) | rejected | contaminated by sparse GT |
| SOT-2848 | naive foreground-weighted MSE (bg weight 1.0 **everywhere**) | **0.0** | every unannotated cell taught as background → degenerate under-firing |
| SOT-2863 | nnPU / Cellsparse soft down-weight (bg 0.05 **everywhere**) | 0.1173 / 0.2753 | curbed but still supervises the whole background → no family-invariant operating point |

All three supervised the **entire** unlabeled background. SOT-2993's masked sparse
loss is a genuinely different mechanism: it takes the annotation sparsity literally
and supervises **only a bounded neighbourhood of each GT annotation**, excluding the
rest of the volume from the loss entirely.

## Mechanism — masked sparse supervision

`src/biohub_tracking/learned_detect.py` (pure-numpy, unit-tested):

- `gaussian_heatmap_target(shape, points, sigma_zyx)` — sum-of-Gaussians target
  (peak 1.0) at each sparse GT centre.
- `annotation_supervision_mask(shape, points, radius_zyx)` — boolean **supervised
  field of view**: `True` only within an anisotropic ellipsoid (`radius_zyx=[4,12,12]`
  voxels ≈ 4·σ) of any GT point.
- `masked_sparse_loss_weights(...)` — per-voxel loss weight: `fg_weight` at blob
  cores, `1.0` at local background **inside the mask**, and **exactly 0 outside**.

The training loss is `(w·(sigmoid(logit) − target)²).sum() / w.sum()`. Because
`w == 0` outside the mask, an unannotated real cell far from any GT annotation
contributes **no gradient** and is never pushed toward background — the precise PU
contamination cure. This is pinned by `tests/test_learned_detect_masked.py`
(`test_unannotated_cell_is_excluded_from_loss`, and the torch
`test_masked_loss_gradient_zero_outside_mask` proving `grad[~mask] == 0`).

## Experiment harness (same-seed, detection-only swap)

Byte-identical to SOT-2863's harness (same families, GT-centred patch sampler, arch,
seed) so the A/B isolates the loss as the single variable:

- `experiments/sot2993/train_lofo.py` — trains 4 leave-one-family-out folds
  (`weights/masked/<family>.pt`), each on the other three families' sparse GT.
- `experiments/sot2993/run_ab.py` — scores the `masked` arm through the leak-free CV
  primitives (`biohub_tracking.eval.cv`), champion linker fixed, per-dataset
  non-regression gate. Committed operating point threshold 0.5 + diagnostic sweep
  {0.3, 0.5, 0.7}.

Reproduce (repo `.venv`, GPU):

```bash
.venv/bin/python experiments/sot2993/train_lofo.py --steps 1500 --device cuda
.venv/bin/python experiments/sot2993/run_ab.py --arms masked --device cuda
```

## Result

**Decision: REJECT (non-promotion).** Same-seed leak-free CV A/B
(`experiments/sot2993/screen_masked_ab.json`, 4 LOFO folds, champion linker fixed,
committed operating point threshold 0.5):

| family | champion adj | masked adj | Δ |
| --- | --- | --- | --- |
| 44b6_0113de3b | 0.9078 | 0.8964 | −0.0114 |
| 44b6_0b24845f | 0.6938 | 0.4706 | −0.2232 |
| 6bba_05b6850b | 0.5748 | 0.5727 | −0.0021 |
| 6bba_05db0fb1 | 0.7477 | 0.6569 | −0.0908 |
| **micro-adj** | **0.6760** | **0.6217** | **−0.0543** |

`micro_adj_delta = −0.0543` (well below the −0.003 noise band) and `no_regression =
False` (every family regresses) ⇒ gate = **REJECT**.

**What the masked loss *did* fix.** The mask is a genuine cure for the sparse-GT PU
contamination that degenerated the earlier learned detectors: masked-sparse reaches
micro-adj **0.6217**, vs SOT-2848 naive-MSE **0.0** (degenerate under-firing) and
SOT-2863 nnPU/Cellsparse **0.1173 / 0.2753**. So excluding the unlabeled background
from the loss (verified in `tests/test_learned_detect_masked.py`:
`grad[~mask] == 0`) is the right mechanism and lifts the learned detector to within
~0.05 of the classical champion for the first time.

**Why it still loses.** Even with the PU cure, the self-trained U-Net does not reach
a family-invariant operating point that beats the byte-frozen classical DoG+NMS
champion (0.6760). The loss concentrates on 44b6_0b24845f (−0.2232) and 6bba_05db0fb1
(−0.0908); the diagnostic threshold sweep {0.3, 0.5, 0.7} does not recover a single
threshold that non-regresses (e.g. thr 0.3 collapses 6bba_05b6850b to 0.0; thr 0.7
collapses 6bba_05db0fb1 to 0.1428). The classical champion stays champion (byte-frozen,
sha256 unchanged); the learned arm ships **default-off** as documented evidence only.
No Kaggle submission from this child.

The learned **detector** axis is now exhausted three-mechanisms-deep (re-rank scorer
→ naive MSE → PU-downweight → masked-sparse); the remaining head-room is not in
punishing the classical detector's operating point but in a fundamentally stronger
detection substrate (frontier/official baseline territory), which is beyond a single
default-off child. Recorded for the parent cycle's escalation ladder.

## Offline exec-compat

`.venv/bin/python -m biohub_tracking.learned_detect --smoke` confirms the offline
attach → load → forward path (`resolved_offline: true`, `state_dict_bit_exact: true`,
`forward_ok: true`, GPU). Inference is `__file__`-independent and cwd-independent:
`LearnedDetector.resolve_weights_path` discovers the checkpoint under the attached
Kaggle Dataset mount `/kaggle/input/…` with no network (SOT-2847 verdict). The
candidate config `champion/candidates/sot2993-learned-masked-detect.json` references
the weights at `biohub-claude-weights-sot2993/detector.pt`.

## Disposition

Default-off; classical champion `champion/config.json` byte-frozen (sha256
unchanged), preserved as the hedge. No Kaggle submission from this child.
