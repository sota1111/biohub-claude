# SOT-1983 — Detection + Linking baseline champion

First **champion** for biohub-claude: a classical (numpy/scipy only) 3D cell
**detection** + frame-to-frame **linking** pipeline, measured with the local
Edge/Division Jaccard evaluator built in SOT-1982 and promoted through a
**screen → confirm** gate.

## Data

- **Image** `data/test/44b6_0113de3b.zarr` — OME-NGFF `(T=100, Z=64, Y=256,
  X=256)` uint16 light-sheet volume, reconstructed from Kaggle (chunked
  1-per-timepoint, ~437 MB). Not redistributable (gitignored).
- **Ground truth** `data/train/44b6_0113de3b.geff` — the one released annotation:
  **52 nodes / 50 edges, a single cell lineage** (one node per timepoint, a gap
  t=3..26), **no divisions**. Estimated true node count ≈ 25 755 (the GT is
  deliberately sparse).
- Physical voxel scale `(z, y, x) = (1.625, 0.40625, 0.40625)` µm; all matching
  is in microns, 7 µm gate.

## Method

**Detection** (`biohub_tracking.detect`), per timepoint:
1. Anisotropic Gaussian smoothing `sigma=(1, 3, 3)` (small along the coarse Z
   axis) to suppress shot noise while keeping nuclei compact.
2. Non-max-suppressed local maxima (`maximum_filter`, footprint `(5, 11, 11)`),
   so each nucleus yields one centroid.
3. Keep maxima above an **adaptive percentile threshold** of the smoothed
   volume (image-adaptive — survives the strong intensity drift across the
   embryo).

**Linking** (`biohub_tracking.link`): optimal (Hungarian) nearest-neighbour
assignment of centroids between consecutive frames within 7 µm; out-of-range
pairs rejected (tracks start/end freely). Division handling (attach a second
daughter as a `1 → 2` split) is implemented but **disabled** in the champion —
see below.

Everything is deterministic (no RNG) → scores reproduce exactly.

## Detection recall (sanity)

The detector places a centroid within the 7 µm gate of the GT cell at **51 / 52**
timepoints (median nearest distance 0.41 µm; the one miss is 8.1 µm). Node recall
is therefore not the bottleneck; edge quality is set by linking.

## Screen → confirm gate

Metric: size-weighted **adjusted edge Jaccard** `+ 0.1 × division Jaccard`
(the leaderboard metric). Full numbers in
[`baseline-evaluation.json`](baseline-evaluation.json).

**Screen** (threshold × division on/off):

| thr % | div | nodes | edge TP/FP/FN | adj edge J | score |
| --- | --- | --- | --- | --- | --- |
| 99.0 | on  | 13958 | 47/3/3 | 0.9274 | 0.9274 |
| 99.0 | off | 13958 | 47/2/3 | 0.9452 | 0.9452 |
| 99.3 | on  | 12269 | 47/3/3 | 0.9332 | 0.9332 |
| **99.3** | **off** | **12269** | **47/2/3** | **0.9512** | **0.9512** |
| 99.5 | on  | 10924 | 45/5/5 | 0.8653 | 0.8653 |
| 99.5 | off | 10924 | 45/2/5 | 0.9152 | 0.9152 |
| 99.7 | off | 9011  | 42/1/8 | 0.8771 | 0.8771 |

Division handling **always loses**: the GT sample has no divisions, so every
predicted split is a false positive (division FP, and extra edge FP). → disabled.

**Confirm** (finer threshold sweep, division off): edge TP/FP/FN is a **flat
plateau 47/2/3 across thr 99.0–99.4** (identical matching quality); the adjusted
score rises only via the node-count penalty term as fewer nodes are kept, until
recall falls off a cliff at 99.5 (TP 47→45). The pipeline is **reproducible**
(identical on a second run). thr = 99.3 is chosen: the coarse-grid argmax, mid
plateau, with margin from the 99.5 recall cliff — not a knife-edge maximum.

## Result — PROMOTED

Champion `detect-link-v1` registered in [`../registry.json`](../registry.json) /
[`../champion/config.json`](../champion/config.json):

- **adjusted edge Jaccard = 0.9512**, edge TP/FP/FN = 47/2/3, division Jaccard =
  n/a (no GT divisions in this sample), on `44b6_0113de3b`.
- Config: detect `thr=99.3`, `sigma=(1,3,3)`, `nms=(2,5,5)`; link `max_dist=7 µm`,
  `allow_division=false`.

## Caveats / next levers

- **Single local GT sample** (one lineage, no divisions) — the score confirms the
  pipeline works end-to-end and is reproducible, but it is one point. The adjusted
  term depends on `estimated_number_of_nodes` (25 755), which differs per dataset.
- **Divisions forfeited.** The real test set has divisions worth `0.1 ×` division
  Jaccard. A future issue should add a proper division detector (the code path
  exists behind `allow_division`) and re-run the gate.
- The detector under-detects vs the ~25 755 estimate (keeps ~12 k of the brightest
  peaks); that *helps* the adjusted term here but likely hurts recall on denser
  datasets — a threshold/scale study is the next lever.

## Reproduce

```bash
pip install -e .
# reconstruct data/test/44b6_0113de3b.zarr from Kaggle first
python scripts/evaluate_baseline.py                                   # regenerates baseline-evaluation.json
python -m biohub_tracking.build_submission --test-dir data/test --out submission.csv
biohub-evaluate --pred submission.csv --gt-dir data/train             # 0.9512 adj edge J
```
