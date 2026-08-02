# Champion — `detect-link-dog-v3-adaptive`

The reigning detection + linking configuration for biohub-claude. Established in
**SOT-1983** (`detect-link-v1`), superseded in **SOT-2272** by `detect-link-dog-v2`
(local-contrast Difference-of-Gaussians so dim tracked cells are found), and
superseded in **SOT-2307** by `detect-link-dog-v3-adaptive`, which swaps the fixed
percentile-92 response cutoff for a **robust per-volume adaptive threshold**
(`median + 3.0·1.4826·MAD` of the DoG response) so the detection *count* adapts to
each dataset's own noise floor instead of keeping a fixed voxel fraction. State is
stored declaratively so promotion is data, not code:

- [`config.json`](config.json) — the frozen champion parameters.
- [`../registry.json`](../registry.json) — champion pointer + headline metrics + gate.
- [`../docs/ai/sot-2307-adaptive-normalization.md`](../docs/ai/sot-2307-adaptive-normalization.md) — current champion method + screen→confirm writeup.
- [`../experiments/sot2307/confirm_adaptive.json`](../experiments/sot2307/confirm_adaptive.json) — machine-readable confirm scores.
- [`../docs/sot-2272-dog-detection.md`](../docs/sot-2272-dog-detection.md) — prior DoG-v2 champion writeup.
- [`../docs/sot-1983-baseline.md`](../docs/sot-1983-baseline.md) — original v1 baseline.

## What it does

1. **Detection** (`biohub_tracking.detect`) — per timepoint, anisotropic Gaussian
   smoothing → a **Difference-of-Gaussians** local-contrast response
   (`gaussian(σ_small) − gaussian(σ_background)`) → non-max-suppressed local
   maxima above a **robust per-volume threshold** (`median + 3.0·1.4826·MAD` of the
   response, SOT-2307) → cell centroids (voxel coords). DoG detects cells that are
   brighter than their local surround even when they are globally dim, recovering
   the tracked cell in the fragmented family `44b6_0b24845f` (~p60 of the smoothed
   volume) that a brightness threshold missed; the robust MAD cutoff then keeps the
   kept-peak count proportional to each dataset's signal content rather than a fixed
   voxel fraction, stopping the sparse-dataset over-detection (`6bba_05b6850b`
   pred 40450→13376) that a fixed percentile caused.
2. **Linking** (`biohub_tracking.link`) — optimal (Hungarian) nearest-neighbour
   assignment between consecutive frames within 7 µm; division handling
   **disabled** in the champion.

## Reproduce

```bash
pip install -e .
# 1. download + reconstruct data/test/44b6_0113de3b.zarr from Kaggle (see the eval script)
python scripts/evaluate_baseline.py          # regenerates docs/baseline-evaluation.json
# build a submission with the champion over a dir of test videos:
python -m biohub_tracking.build_submission --test-dir data/test --out submission.csv
```

Score over the SOT-2305 **4-dataset LB holdout** (`44b6_0113de3b`, `44b6_0b24845f`,
`6bba_05b6850b`, `6bba_05db0fb1`): **micro-averaged adjusted edge Jaccard = 0.6232**
(up from the DoG-v2 incumbent's 0.5225; the 2-family 44b6 micro is held at 0.7716 vs
0.7721 while the sparse 6bba prefix rises 0.5117→0.6168). Deterministic — no RNG.
The clean family keeps its 47/2/3 edges; the fragmented family rises from edge adj
0.044 to 0.662.

## Promotion rule

A challenger replaces the champion only if it **beats these metrics under the
same screen→confirm gate**. If it does not, revert the code and record the result
in `docs/` — do not pollute the champion.
