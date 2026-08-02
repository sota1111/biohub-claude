# SOT-2272 — DoG detection recovers dim tracked cells (champion `detect-link-dog-v2`)

## Summary

The public leaderboard had been stuck at **0.509** across the previous improvement
cycles, which had concluded the score was "measurement-blocked" (no local oracle).
That conclusion was **wrong**. This cycle established a valid local oracle, found
the exact source of the loss, fixed it, and promoted a new champion.

- **Local oracle found.** Scoring the champion over the *two* GT families we hold
  locally — `44b6_0113de3b` (clean) and `44b6_0b24845f` (fragmented) — gives a
  micro-averaged adjusted edge Jaccard of **0.5063**, which reproduces the public
  LB **0.509**. So the local 2-family eval is a faithful promotion oracle.
- **Loss localized.** The clean family scores adj **0.9512** (47/2/3 edges). The
  fragmented family scores adj **0.0436** — only **2 of 49** GT edges recovered.
  All of the loss lives there.
- **Root cause.** The single tracked cell in `44b6_0b24845f` is *dim*: it sits at
  ~**p60** of the Gaussian-smoothed volume (raw ~1400–1700 vs frame p99.3 ~2200–2720).
  The champion's global intensity-percentile threshold (99.3) keeps only the top
  0.7% brightest peaks and never sees it. Lowering the global threshold does **not**
  help — even at p70 the fragmented family's edge TP is stuck at ≤5, while the clean
  family degrades via the node-count penalty (dead end, confirmed by a full sweep).
- **Fix.** Detect by **local contrast**, not absolute brightness. A
  Difference-of-Gaussians response `gaussian(σ_small) − gaussian(σ_large)` finds the
  cell as a compact blob brighter than its immediate surround. DoG detection recall
  on the dim cell jumps from **8–18% → 92%**.

## Champion `detect-link-dog-v2`

```
detect: sigma_zyx=[1,2,2]  background_sigma_zyx=[2,6,6]  nms=[2,5,5]  threshold_percentile=92.0
link:   max_distance=7µm   allow_division=false
```

The only change vs `detect-link-v1` is the **detection response**: when
`background_sigma_zyx` is set the detector thresholds the DoG response instead of
raw smoothed intensity (`biohub_tracking.detect.DetectParams.background_sigma_zyx`,
backward-compatible — `None` reproduces v1 exactly). Linking is unchanged.

## Screen → confirm

| detector | frag edge TP/FP/FN | frag adj | clean adj | micro-adj |
| --- | --- | --- | --- | --- |
| v1 intensity p99.3 (incumbent) | 2/1/47 | 0.044 | 0.951 | **0.506** |
| intensity p95 (best global thr) | 5/3/44 | 0.103 | 0.917 | 0.510 |
| top-K/frame (best) | 7/… | — | — | 0.525 |
| **DoG (1,2,2)/(2,6,6) p92** | **38/5/11** | **0.662** | **0.887** | **0.772** |

DoG `threshold_percentile` sweep (full edge score, both families):
p90=0.738, **p92=0.772**, p93=0.750, p94=0.761, p95=0.755, p96=0.760, p97=0.739 →
**p92 chosen**. The clean family keeps its 47/2/3 edges at every setting, so this is
a detection-only, contrast-based gain — not an overfit threshold.

**Local micro-avg: 0.5063 → 0.7722 (+0.266).** Promotion is on the measured oracle;
the two unseen test families (`6bba_*`) are also fragmented, so DoG's local-contrast
detection is expected to help there as well, but that is unverified until the LB
returns. Reproduce with `experiments/sot2272/screen_dog.py` (see the evaluation JSON
`sot-2272-dog-detection-evaluation.json`).
