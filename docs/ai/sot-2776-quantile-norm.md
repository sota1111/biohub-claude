# SOT-2776 — per-volume robust quantile intensity normalization before DoG (REJECTED)

**Cycle 3 (SOT-2773).** Axis: prepend a per-volume `[plow, phigh]` percentile
clip → `[0, 1]` rescale (royerlab's public-baseline contrast stretch) to the raw
volume before Gaussian smoothing, hoping to stabilise the adaptive-MAD threshold
across the embryo's brightness drift and recover dim family/timepoint cells
(lower edge FN).

## Result: REJECTED — every variant regresses the leak-free CV, no dim-cell recall gain

Screen: `experiments/sot2776/screen_quantile_norm.py` →
`experiments/sot2776/screen_quantile_norm.json`. Scored through the one
leak-free CV evaluator (`biohub_tracking.eval.cv`, SOT-2761). Downstream is the
frozen champion (DoG-v3 adaptive `mad_k=3.0` + short-track `min_track_length=4`);
only `DetectParams.intensity_norm` varies. Detection is re-run per variant
because normalization changes the response; the MAD threshold is recomputed on
the post-normalization response automatically.

`baseline_none` reproduces the registry champion **0.6649** exactly (same-seed
A/B basis; the pipeline is deterministic).

| variant | micro-adj | Δ vs 0.6649 | per-dataset no-regression |
| --- | --- | --- | --- |
| baseline_none (champion) | **0.6649** | — | — |
| q_p1_p99.9 | 0.6511 | −0.0138 | no |
| q_p0.5_p99.5 | 0.6159 | −0.0490 | no |
| q_p1_p99 | 0.5887 | −0.0762 | no |
| q_p2_p98 | 0.5451 | −0.1198 | no |
| q_p5_p95 | 0.5138 | −0.1511 | no |

Monotonic: the harder the percentile clip, the worse the score. **No variant
passes the gate** (micro improvement AND per-dataset no-regression); the least
aggressive band still both drops the micro and regresses every family.

## Why the hypothesis is wrong — recall got *worse*, not better

The stated motivation was that dim cells are missed and normalization would
recover them (fewer FN). The A/B shows the opposite. Under `q_p1_p99` on the
sparse/dim families:

- `6bba_05b6850b`: edge **TP 651→528, FN 194→317, FP 215→307** — recall fell.
- `44b6_0113de3b`: 47/2/3 → 44/10/6 (clean family degraded too).

Root cause: the detector already handles brightness drift the *right* way. It
finds peaks by **local contrast** (Difference-of-Gaussians), not absolute
brightness, and thresholds with a per-volume robust **MAD z-score**
(`median + 3·1.4826·MAD`) that is already scale-adaptive. A global per-volume
percentile clip adds nothing those two steps don't already do, and actively
hurts: clipping to `[p1, p99]` **compresses the dynamic range** and saturates the
bright tail where real nuclei live, so the DoG response contrast between a dim
nucleus and its surround *shrinks*, and the MAD threshold on the flattened
response admits more noise (FP↑) while dropping genuine dim peaks (FN↑). The one
family that barely improved (`6bba_05db0fb1`, dense) does not offset the sparse-
family collapse.

Intensity normalization is a lever for a raw-brightness detector; this repo
already moved past that (SOT-2272 DoG, SOT-2307 adaptive MAD). No confirm run is
warranted — the screen has no promotion candidate to confirm.

## Disposition

- **Champion unchanged.** `champion/config.json` / `registry.json` untouched;
  `EMBEDDED_CHAMPION_CONFIG` untouched; exec-compat gate green.
- **Knob kept, default-off.** `DetectParams.intensity_norm=None` reproduces the
  pre-SOT-2776 detector byte-for-byte (unit-tested); `champion_params()` reads an
  optional `detect.intensity_norm` only if a future config sets one. Kept as an
  inert, tested building block (same pattern as SOT-2762's default-off
  `division_max_sibling_ratio`) rather than deleted.
- **No Kaggle submission** (submission is the parent Issue's job).

Next axes should target the actual score ceiling (matching/linking on the sparse
families, or detection-stage escalation SOT-2773), not global intensity rescaling.
