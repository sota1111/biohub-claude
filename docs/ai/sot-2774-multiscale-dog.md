# SOT-2774 — Multi-scale scale-space DoG/LoG blob detection (REJECTED, default-off)

**Cycle 3 (biohub-claude Kaggle rank cycle #2). Axis: detection stage.**
**Verdict: REJECTED.** Champion `detect-link-dog-v4-shorttrack` maintained (CV micro-adj **0.6649**).
Kept as a documented default-off `DetectParams.dog_scales_zyx` knob (`None` reproduces the single-scale
champion bit-for-bit). No Kaggle submission.

## Hypothesis

The champion detector is a **single-scale** DoG (foreground σ `[1,2,2]` vox, background `[2,6,6]`) +
adaptive-MAD threshold + NMS local-maxima. It is tuned to one blob radius. Real nuclei vary in size
across family/timepoint (sparse large `44b6` vs dense small `6bba`), so one scale should both miss large
cells and over-split small ones. A **multi-scale scale-space DoG** — per-voxel max of the
normalized-LoG-corrected (`σ̄² = prod(σ)^(2/3)`) DoG response over a small σ bank, numpy/scipy only, no
new dependency — should improve the detection recall/precision balance across families without GPU
weights.

## Method

- New `DetectParams.dog_scales_zyx: tuple[(σz,σy,σx), ...] | None` (default `None` = exact single-scale
  reproduction). `_multiscale_dog_response` computes a DoG per scale (each scale's background σ scaled by
  the base `background/sigma` ratio, so the middle scale `== sigma_zyx` reproduces the champion response),
  multiplies by the normalized-LoG `σ̄²` correction, and keeps the per-voxel max (+ argmax-scale index for
  diagnostics). The combined response then feeds the **same** MAD z-score threshold + NMS path unchanged.
  Deterministic (no RNG), exec-compatible.
- Screen: single-variable same-seed A/B on the **SOT-2761 leak-free 4-family CV** with detection as the
  variable and the frozen champion downstream (DoG-v4 adaptive mad_k=3.0 + short-track mtl=4). Gate =
  micro-adj improvement **AND** per-dataset no-regression vs `baseline_none`.
  Script `experiments/sot2774/screen_multiscale.py`, results `experiments/sot2774/screen_multiscale.json`.

## Result — no safe operating point

`baseline_none` reproduces the champion **0.6649** exactly (wiring sanity).

| variant | scales | micro-adj | Δ vs base | per-dataset no-regression | gate |
| --- | --- | --- | --- | --- | --- |
| bank3_center | (0.7,1.4,1.4)/(1,2,2)/(1.4,2.8,2.8) | 0.6859 | **+0.0210** | **False** | ✗ |
| bank3_coarse | (1,2,2)/(1.4,2.8,2.8)/(2,4,4) | 0.6775 | +0.0126 | False | ✗ |
| bank5_wide | 5-scale ladder | 0.6495 | −0.0154 | False | ✗ |
| bank3_fine | (0.6,1.2,1.2)/(0.85,1.7,1.7)/(1.1,2.2,2.2) | 0.6338 | −0.0311 | False | ✗ |

The two micro-positive banks buy their gain on the dense small-nuclei family at the cost of the sparse
family, so **no bank passes the per-dataset no-regression gate**:

- **bank3_center** (best micro +0.021): dense `6bba` improves (`6bba_05b6850b` adj 0.5700→0.6460, TP
  651→705 / FP 215→89), but sparse `44b6` **regresses** — `44b6_0113de3b` 0.8895→0.8631 and
  `44b6_0b24845f` 0.6817→**0.5583** (TP 37→30, FN 12→19). Adding fine scales inflates `44b6` pred-node
  count (29844→37358) and sprays extra centroids that steal the per-timepoint ≤7µm matches, turning real
  edges into FN.
- **bank3_coarse**: `6bba_05b6850b` 0.5700→**0.7599** (FP 215→46) but the densest `6bba_05db0fb1`
  0.7310→0.6213 (FN 210→382, the coarse-only bank blurs adjacent nuclei together) and sparse
  `44b6_0b24845f` 0.6817→0.4665.
- Fine/wide banks regress micro outright (fine scales dominate the max on `6bba` noise → FP up).

**Root cause.** Multi-scale max is a *union* of blob radii, so it monotonically raises the detected-count.
On the dense family that recovers fused/size-varied nuclei (good); on the sparse `44b6` family (few real
cells, sparse GT that penalises over-count) the extra scales only add false/duplicate centroids and steal
matches. The single-scale champion is already near the per-family Pareto point; the `σ̄²` normalization
makes the scales comparable but cannot make one global bank that is simultaneously additive-good on dense
and non-additive on sparse. This is the **same family-balance impossibility** seen for watershed splitting
(SOT-2775) and quantile normalization (SOT-2776): a single global detection knob has no operating point
that helps the dense family without hurting the sparse one.

## Decision

- Champion **maintained** (`detect-link-dog-v4-shorttrack`, CV micro-adj 0.6649). `champion/config.json`,
  `registry.json`, and `EMBEDDED_CHAMPION_CONFIG` byte-unchanged (`dog_scales_zyx` absent → `None`).
- `dog_scales_zyx` kept as a **documented default-off** `DetectParams` knob (like SOT-2369 `velocity_gain`,
  SOT-2762 `division_max_sibling_ratio`, SOT-2763 `max_frame_gap`, SOT-2776 `intensity_norm`,
  SOT-2775 `watershed`). `None` reproduces the single-scale detector bit-for-bit
  (`tests/test_multiscale_dog.py::test_none_reproduces_single_scale_exactly`).
- pytest (87) + exec-compat gate green. No Kaggle submission.

## Next axis

Detection knobs that act as a single global recall lever are exhausted on this CV (multi-scale, watershed
split, quantile norm all rejected — each trades dense-family gain for sparse-family loss). The remaining
portable direction is a **per-family / per-density adaptive** detector (e.g. condition the bank or
threshold on local nucleus density) rather than one global setting, or accept the 0.6649 detection ceiling
and escalate the ladder toward the non-portable frontier (GPU UNet-transformer) as out-of-scope.
