# SOT-2884 — Ultrack multi-hypothesis detection selection by temporal support

**Cycle 8 · biohub-claude · axis: detection-stage re-anchor · Verdict: REJECTED (champion byte-frozen)**

## Mechanism (default-off, `detect_hypothesis_select`)

Ports Ultrack's core principle — *do not commit early to one segmentation; keep a
pool of candidate hypotheses and let temporal consistency choose the final
mutually-exclusive set* — to the point-detector's **detection stage**:

1. **Threshold-ladder candidate pool** (`detect.candidate_pool`): per frame, keep the
   NMS local maxima of the champion DoG response above the *lower* robust-z gate
   `median + hypothesis_mad_k_low·1.4826·MAD` — a strict **superset** of the
   champion `mad_k=3.0` peaks.
2. **Pool linking** (`pipeline._hypothesis_select_detections`): link the whole pool
   with the champion motion cost but **without** short-track pruning / gap-recovery /
   division overlay, so each candidate's weakly-connected-component size *is* its
   **temporal support** (the length of the motion-consistent track it earns).
3. **Selection**: keep only candidates with support `>= hypothesis_min_track (L)`;
   re-link the surviving disjoint detections with the champion linker.

**Distinct from the rejected detection axes:** a threshold *ladder resolved by
cross-frame support* (not the single fused operating point of SOT-2774 multiscale-DoG),
and sub-threshold candidates admitted **only when temporally supported** (not the
bounded *unconditional* sub-threshold add of SOT-2873 recall-recovery).

## Leak-free 4-family LOFO A/B (same seed, vs byte-frozen champion micro-adj 0.6649)

| variant (mad_k_low, L) | micro-adj | non-regressing? | 44b6_0113de3b | 44b6_0b24845f | 6bba_05b6850b | 6bba_05db0fb1 |
|---|---|---|---|---|---|---|
| **(3.0, 4)** | 0.6649 | **YES** | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| (2.5, 3) / (2.5, 4) | 0.6652 | no | −0.0053 | +0.0066 | −0.0169 | +0.0140 |
| (3.0, 5) | 0.6691 | no | −0.0356 | +0.0053 | +0.0087 | +0.0024 |
| (2.5, 5) | 0.6714 | no | −0.0397 | +0.0128 | −0.0065 | +0.0187 |
| (2.0, 5) | 0.6586 | no | −0.0500 | −0.0224 | −0.0367 | +0.0200 |

(adjusted edge-Jaccard delta vs champion; per-family TP/FP/FN in `docs/ai/sot2884/ab_results.json`.)

## Verdict: REJECTED

- The **only** non-regressing operating point, `(3.0, 4)`, exactly reproduces the
  champion (pool = champion peaks, support ≈ mtl=4 prune) — **zero gain**.
- **Every** micro-gaining variant regresses the precision-saturated clean sparse
  family **44b6_0113de3b** (champion adj 0.8895, FP=2), in *both* directions:
  - low L / low gate → **over-prediction penalty**: identical TP/FP/FN (47/2/3) but
    +1500 pred_nodes drops adj −0.0053 (the metric's `J·(1−0.1·(N_pred−N_true)/N_true)`).
  - high L → **real short-track loss**: TP 47→45, adj −0.0356…−0.0500.
- The axis genuinely recovers the under-detected dense **6bba_05db0fb1** (+0.014…+0.020,
  TP +19…+31), but that gain is inseparable from the 44b6_0113de3b regression — the
  **same family-mix wall** the champion hit when it chose `min_track_length=4` over
  `mtl=5`. The mandatory per-dataset non-regression gate is not cleared.

Detection stage remains saturated (cf. SOT-2863 sparse-GT PU contamination, SOT-2773
detection-exhausted). Champion `detect-link-dog-v4-shorttrack` stays **byte-frozen**;
`detect_hypothesis_select` stays **default-off** (champion config carries no key;
`eval.cv --check-champion` reproduces 0.6649 with the code in place).

## Verification

- `tests/test_hypothesis_select.py` — 10 tests (byte-invariance of the per-frame
  detector, pool superset property, cap, temporal-support keep/drop, config plumbing).
- Full suite **212 passed**; `submit/exec_compat_gate.py` OK; `compileall` OK.
- **Kaggle NOT submitted** (byte-frozen champion is fingerprint-identical to the last
  submission; no new artifact).
