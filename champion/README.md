# Champion — `detect-link-dog-v2`

The reigning detection + linking configuration for biohub-claude. Established in
**SOT-1983** (`detect-link-v1`) and superseded in **SOT-2272** by
`detect-link-dog-v2`, which swaps the detection response for a local-contrast
Difference-of-Gaussians so dim tracked cells are found. State is stored
declaratively so promotion is data, not code:

- [`config.json`](config.json) — the frozen champion parameters.
- [`../registry.json`](../registry.json) — champion pointer + headline metrics + gate.
- [`../docs/sot-2272-dog-detection.md`](../docs/sot-2272-dog-detection.md) — current champion method + screen→confirm writeup.
- [`../docs/sot-2272-dog-detection-evaluation.json`](../docs/sot-2272-dog-detection-evaluation.json) — machine-readable scores.
- [`../docs/sot-1983-baseline.md`](../docs/sot-1983-baseline.md) — original v1 baseline.

## What it does

1. **Detection** (`biohub_tracking.detect`) — per timepoint, anisotropic Gaussian
   smoothing → a **Difference-of-Gaussians** local-contrast response
   (`gaussian(σ_small) − gaussian(σ_background)`) → non-max-suppressed local
   maxima above an adaptive percentile threshold → cell centroids (voxel coords).
   DoG detects cells that are brighter than their local surround even when they
   are globally dim, recovering the tracked cell in the fragmented family
   `44b6_0b24845f` (~p60 of the smoothed volume) that a brightness threshold missed.
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

Score over the two local GT families `44b6_0113de3b` (clean) + `44b6_0b24845f`
(fragmented): **micro-averaged adjusted edge Jaccard = 0.7722** (up from the v1
incumbent's 0.5063, which reproduces the public LB 0.509). Deterministic — no RNG.
The clean family keeps its 47/2/3 edges; the fragmented family rises from edge adj
0.044 to 0.662.

## Promotion rule

A challenger replaces the champion only if it **beats these metrics under the
same screen→confirm gate**. If it does not, revert the code and record the result
in `docs/` — do not pollute the champion.
