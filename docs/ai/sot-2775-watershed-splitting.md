# SOT-2775 — watershed nucleus splitting (h-maxima seeding) for dense families (REJECTED)

**Cycle 3 (SOT-2773).** Axis: the champion detector reports one centroid per NMS
local maximum, so two nuclei whose blurred DoG blobs merge into a single response
peak collapse to one detection — the matched-edge FN/FP source on the *dense*
`6bba` families (`6bba_05b6850b` adj 0.5700 FN194 / `6bba_05db0fb1` adj 0.7310
FN210). Port the classical light-sheet nucleus-splitting recipe (DoG + h-maxima
seeding → marker-controlled watershed, numpy/scipy only) to split fused blobs and
recover dense-family TP.

## Method

New default-off `DetectParams.watershed = ("hmaxima", h, min_size, min_seed_dist)`
(`src/biohub_tracking/detect.py`). When set, the NMS peak extractor (steps 2-3 of
the detector) is replaced by, on the *same* adaptive-MAD-threshold foreground:

1. **h-maxima seeding.** Per connected foreground component, the
   **extended-maxima transform** (regional maxima of the h-maxima transform, via
   `scipy.ndimage` grey reconstruction) yields seeds whose prominence exceeds `h`
   robust-sigma (`h · 1.4826 · MAD` of the response — the same robust scale as
   `mad_k`), so shallow noise maxima do not seed a spurious split.
2. **Marker-controlled split.** The component is partitioned into one basin per
   seed by exact nearest-seed Euclidean assignment (`distance_transform_edt`
   feature indices — the deterministic geometric limit of a marker-controlled
   watershed; it falls at the response ridge between compact nuclei). Each basin's
   centroid is a detection.
3. **Over-split control.** Drop basins below `min_size` voxels; greedily suppress
   centroids within `min_seed_dist` voxels (keep the brightest).

`watershed=None` reproduces the NMS detector byte-for-byte (deterministic,
numpy/scipy-only, Kaggle-kernel safe). Wired through `champion_params`.

## Result: REJECTED — splits recover some dense TP but no single knob avoids a per-dataset collapse

Screen: `experiments/sot2775/screen_watershed.py` →
`experiments/sot2775/screen_watershed.json`. Scored through the one leak-free CV
evaluator (`biohub_tracking.eval.cv`, SOT-2761), detection the variable, downstream
frozen champion (DoG-v3 `mad_k=3.0` + short-track `mtl=4`). `baseline_none`
reproduces the registry champion **0.6649** exactly (same-seed A/B; deterministic).

Surgical best-case variant `ws_h3_ms8_sd4` (`h=3` σ, `min_size=8` vox,
`min_seed_dist=4` vox — the least-aggressive, most over-split-guarded setting):

| dataset | champion adj | watershed adj | edge TP/FP/FN champion → watershed |
| --- | --- | --- | --- |
| 44b6_0113de3b (clean dense) | 0.8895 | **0.9105** ✓ | 47/2/3 → 48/2/2 |
| 44b6_0b24845f (sparse/dim)  | 0.6817 | **0.1079** ✗ | 37/5/12 → 6/9/43 |
| 6bba_05b6850b (dense)       | 0.5700 | **0.5836** ✓ | 651/215/194 → 678/249/167 |
| 6bba_05db0fb1 (densest)     | 0.7310 | **0.5974** ✗ | 973/148/210 → 815/215/368 |
| **micro-adj**               | **0.6649** | **0.5869** | Δ **−0.078**, no-regression **FALSE** |

The aggressive variant `ws_h1_ms3_sd0` (`h=1`, no over-split guard) was dropped
after it regressed *both* 44b6 families (0.8895→0.8691, 0.6817→0.5347) and
inflated the node count so far that linking became pathologically slow
(`experiments/sot2775/screen_watershed_h1partial.log`).

## Why it is rejected — the hypothesis is *partly right*, but no global knob wins

The split **did** recover the fused-nucleus TP it was designed to:
`6bba_05b6850b` edge **TP 651→678, FN 194→167** and the clean dense
`44b6_0113de3b` **TP 47→48** — real matched-edge recovery, not node-count games.

But a single global `(h, min_size, min_seed_dist)` cannot serve all four families:

- **Sparse/dim `44b6_0b24845f` collapses** (0.6817→0.1079, TP 37→6, nodes
  34485→18690). The over-split guards (`min_size=8`, `min_seed_dist=4`) that stop
  fragmentation on the dense families simultaneously *delete* this family's small,
  dim real detections — exactly the low-contrast cells (the tracked cell sits at
  ~p60) that DoG local-contrast recovers and a region-centroid + size gate throws
  away. Loosening the guards (the `h1` variant) instead fragments the clean
  families. The guard is a lever with no safe setting across families.
- **Densest `6bba_05db0fb1` loses 158 TP** (0.7310→0.5974, TP 973→815, FN
  210→368). On the densest tissue the split produces *more* centroids that then
  mismatch the per-timepoint ≤7 µm assignment — resurrected/duplicated centroids
  steal matches, turning real edges into FN. Over-splitting corrupts the matching
  it was meant to help.

So the axis is a genuine trade with no net-positive, no-regression operating
point: what recovers dense-family fusion TP (aggressive splitting) destroys sparse
dim families, and what protects sparse families (size/distance guards) still
corrupts the densest family. Consistent with the cycle-3 sibling findings that the
score is bottlenecked by the *detection–matching interaction* under the
node-count-penalised sparse-GT metric, not by any single classical detection knob.

## Confirm & disposition

`experiments/sot2775/confirm_watershed.py` →
`experiments/sot2775/confirm_watershed.json`: guards that `detect.watershed` flows
through `champion_params` (submission-time path) and that the champion config still
resolves to `watershed=None` (champion detector byte-unchanged), re-scores the
champion baseline to reproduce **0.6649**, and re-scores the challenger through the
real `champion_params(config)->run_pipeline` on one fast family to confirm the
wiring reproduces the screen's 0.9105.

Champion **detect-link-dog-v4-shorttrack MAINTAINED**; `champion/config.json` /
`registry.json` / `EMBEDDED_CHAMPION_CONFIG` byte-unchanged. The watershed path is
kept as a documented default-off `DetectParams` knob (like the SOT-2369
`velocity_gain`, SOT-2762 `division_max_sibling_ratio`, SOT-2763 `max_frame_gap`,
SOT-2776 `intensity_norm` after their rejections). No Kaggle submission.
