# Champion — `detect-link-v1`

The reigning detection + linking configuration for biohub-claude, established in
**SOT-1983** as the first champion. State is stored declaratively so promotion is
data, not code:

- [`config.json`](config.json) — the frozen champion parameters.
- [`../registry.json`](../registry.json) — champion pointer + headline metrics + gate.
- [`../docs/sot-1983-baseline.md`](../docs/sot-1983-baseline.md) — method + screen→confirm writeup.
- [`../docs/baseline-evaluation.json`](../docs/baseline-evaluation.json) — machine-readable scores.

## What it does

1. **Detection** (`biohub_tracking.detect`) — per timepoint, anisotropic Gaussian
   smoothing → non-max-suppressed local maxima above an adaptive percentile
   threshold → cell centroids (voxel coords).
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

Score on the local GT sample `44b6_0113de3b` (52 nodes / 50 edges, single
lineage, no divisions): **adjusted edge Jaccard = 0.9512** (edge TP/FP/FN =
47/2/3). Deterministic — no RNG.

## Promotion rule

A challenger replaces the champion only if it **beats these metrics under the
same screen→confirm gate**. If it does not, revert the code and record the result
in `docs/` — do not pollute the champion.
