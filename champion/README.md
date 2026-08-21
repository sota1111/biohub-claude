# Champion — `detect-link-dog-v4-shorttrack-motion-gain1`

**SOT-2909 (cycle-3 parent-resume) promoted `detect-link-dog-v4-shorttrack-motion-gain1`.**
Same detection as `detect-link-dog-v4-shorttrack`, but the linking stage now runs SOT-2864's
ARGUS-style motion-model predicted-position LAP linking (`motion_model_link=true`,
`motion_smooth_sigma=15.0`, `motion_gain=1.0`, `motion_gate_on_prediction=true`). This is the first
promotion to clear the **two-signal gate**: the exact config was pushed as reserve LB probe
`55662947` and scored **public 0.626** (> the champion's genuine public best 0.624, >> the 0.509
fingerprint-flip artifact) *and* the re-anchored leak-free CV rises **0.6649 → 0.6760** (+0.0111) with
4/4 per-dataset non-regression. Its CV gain therefore demonstrably transfers to the hidden LB,
lifting the SOT-2816 CV↔LB-divergence byte-freeze for this lever. The stronger-CV `motion_gain=2.0`
sibling (`candidates/sot2900-motion-model-link-gain2.json`, CV 0.6821, public unobserved) is HELD as
the #1 reserve for a single converge-phase public check.

## Lineage

The reigning detection + linking configuration for biohub-claude. Established in
**SOT-1983** (`detect-link-v1`), superseded in **SOT-2272** by `detect-link-dog-v2`
(local-contrast Difference-of-Gaussians so dim tracked cells are found), superseded
in **SOT-2307** by `detect-link-dog-v3-adaptive` (robust per-volume `median +
3.0·1.4826·MAD` threshold so the detection *count* adapts to each dataset's noise
floor), and superseded in **SOT-2369** by `detect-link-dog-v4-shorttrack`, which
adds **post-link short-track pruning** (`min_track_length = 4`): after linking, any
weakly-connected track fragment with fewer than 4 nodes is dropped. A real cell
persists across many frames, so a detection that never links into a ≥4-node track
is almost always noise; removing it both relieves the node-count penalty and frees
GT nodes so the ≤7 µm matching attaches the persistent track instead of a transient
decoy. Ported from the public frontier lineage tracker's `FILTER_SHORT_TRACKS`
post-processing (that notebook's 0.913 comes from a GPU pretrained UNet+ILP pipeline
that cannot run under this repo's numpy/scipy/zarr, CPU, no-internet, no-weights
kernel — short-track filtering is the one score lever that transfers). State is
stored declaratively so promotion is data, not code:

- [`config.json`](config.json) — the frozen champion parameters.
- [`../registry.json`](../registry.json) — champion pointer + headline metrics + gate.
- [`../docs/ai/sot-2369-short-track-pruning.md`](../docs/ai/sot-2369-short-track-pruning.md) — current champion method + screen→confirm writeup.
- [`../experiments/sot2369/confirm_shorttrack.json`](../experiments/sot2369/confirm_shorttrack.json) — machine-readable confirm scores.
- [`../docs/ai/sot-2307-adaptive-normalization.md`](../docs/ai/sot-2307-adaptive-normalization.md) — prior DoG-v3-adaptive champion writeup.
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
3. **Short-track pruning** (`biohub_tracking.link`, SOT-2369) — after linking, drop
   every weakly-connected track fragment with fewer than `min_track_length = 4`
   nodes. On the SOT-2305 4-dataset LB holdout this improves all four datasets with
   no regression, lifting the holdout micro-adjusted edge Jaccard 0.6232 → 0.6649
   (+0.042); on the dense `6bba_05b6850b` family matched-edge TP rises 619 → 651
   while FP falls 251 → 215 and FN falls 226 → 194, so the gain is genuine matching
   improvement, not only node-count relief.

## Reproduce

```bash
pip install -e .
# 1. download + reconstruct data/test/44b6_0113de3b.zarr from Kaggle (see the eval script)
python scripts/evaluate_baseline.py          # regenerates docs/baseline-evaluation.json
# build a submission with the champion over a dir of test videos:
python -m biohub_tracking.build_submission --test-dir data/test --out submission.csv
```

Score over the SOT-2305 **4-dataset LB holdout** (`44b6_0113de3b`, `44b6_0b24845f`,
`6bba_05b6850b`, `6bba_05db0fb1`): **micro-averaged adjusted edge Jaccard = 0.6649**
(up from the DoG-v3-adaptive incumbent's 0.6232; every dataset improves with no
regression — 44b6 prefix 0.7716→0.7836, 6bba prefix 0.6168→0.6596). Deterministic —
no RNG. To reproduce the promotion A/B run
`python experiments/sot2369/screen_shorttrack.py` (min_track_length sweep) and
`python experiments/sot2369/confirm_shorttrack.py` (end-to-end champion re-score).

## Promotion rule

A challenger replaces the champion only if it **beats these metrics under the
same screen→confirm gate**. If it does not, revert the code and record the result
in `docs/` — do not pollute the champion.
