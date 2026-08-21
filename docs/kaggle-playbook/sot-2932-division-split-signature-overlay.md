# SOT-2932 — decoupled division-event overlay (image split-signature)

**Explore direction D — harvest the forgone `0.1 · division_jaccard`.** The official
metric is `adjusted_edge_jaccard + 0.1 · division_jaccard`. The champion
(`detect-link-dog-v4-shorttrack-motion-gain1`) runs `link.allow_division=false` and
predicts **zero forks**, so `division_tp = division_fp = 0` and the division Jaccard
is `0.0` — the `0.1` term is entirely forgone. This axis tries to recover it with a
**detector decoupled from the linking cost**, applied as an additive, non-destructive
post-processing overlay.

## Structural difference vs SOT-2762 / SOT-2818 / SOT-2898 (new formulation)

| Attempt | Mechanism | Signal | Verdict |
| --- | --- | --- | --- |
| SOT-2762 | division ON **inside** the linking LAP (`allow_division`) | re-runs assignment | REJECTED (fork FP spray on dense 6bba) |
| SOT-2818 | `nearest-head` overlay | graph **position only** (nearest persistent head) | REJECTED |
| SOT-2898 | `mutual-nn` overlay | graph position + mutual-NN + persistence | REJECTED (tightest gate `div_tp=1` still `+≥4` edge FP, `micro_adj` regressed) |
| **SOT-2932** | `split-signature` overlay | **image intensity condensation + bipolar straddle geometry**, fully decoupled from the LAP | REJECTED (see below) |

The prior overlays are **graph-geometry-only**: they pick a nearby dropped head as the
second daughter using positions alone, and the head they pick is usually *not* the true
daughter, so the added `P → D2` edge is a division FP **and** an edge FP and the edge
component regresses. SOT-2932 keeps the non-destructive additive contract (only *adds*
one `P → D2` edge per fork; OFF / zero-fire ⇒ champion graph byte-for-byte) but proposes
a fork **only on a local strong split signature**:

* **Bipolar straddle geometry (shape).** Unit vectors `u1 = (C−P)/|C−P|`,
  `u2 = (D2−P)/|D2−P|`; the straddle `|u1+u2|` is ~0 for daughters on opposite sides of
  the parent (a real bipolar mitotic split) and ~2 for a co-linear pass-by. Gated by
  `straddle_max`. The prior overlays never tested this — accepting a same-side head is
  the dominant FP mode.
* **Parent condensation (intensity).** `P` must be a bright node in its own frame
  (per-frame champion-DoG response percentile ≥ `parent_bright_pct`): a dividing nucleus
  condenses/brightens before cytokinesis.
* **Both-daughter blobness (intensity).** `C` and `D2` must each be genuine blobs
  (per-frame percentile ≥ `daughter_bright_pct`).

Image evidence enters only through `node_response` (the champion DoG response sampled at
each node voxel, read once per family in the A/B harness); the linking LAP is never
re-run. Params tuple:
`("split-signature", max_distance, sibling_ratio, min_daughter_len, require_parent_track, require_primary_persist, mutual_margin, straddle_max, parent_bright_pct, daughter_bright_pct)`.

## Leak-free same-seed A/B result (cycle 6) — REJECTED

`experiments/sot2932/ab_split_signature.py` → `experiments/sot2932/ab_split_signature.json`.
Baseline reproduces the live champion `micro_adj = 0.6760`, `division_jaccard = 0.0`,
edge FP total 350. Component split recorded per-dataset (edge non-regression vs division
delta). **New evidence: `division_tp = 0` across the entire candidate grid.**

| candidate | div_tp | div_fp | dScore | dMicro | edge_fp | edge non-reg |
| --- | --- | --- | --- | --- | --- | --- |
| geom_str0.8 | 0 | 7 | −0.0006 | −0.0006 | 350→355 | ✗ |
| geom_str0.6 | 0 | 4 | −0.0004 | −0.0004 | 350→353 | ✗ |
| sig_str0.8_p0.6_d0.5 | 0 | 4 | −0.0004 | −0.0004 | 350→353 | ✗ |
| sig_str0.7_p0.7_d0.6 | 0 | 0 | +0.0000 | +0.0000 | 350→350 | ✓ (zero-fire) |
| sig_str0.6_p0.8_d0.7 | 0 | 0 | +0.0000 | +0.0000 | 350→350 | ✓ (zero-fire) |
| sig_str0.5_p0.85_d0.8 | 0 | 0 | +0.0000 | +0.0000 | 350→350 | ✓ (zero-fire) |

**Finding.** The image split-signature does not separate a single true division either:
whenever the gate is loose enough to fire, it produces only FP forks (edge regression);
tightening the intensity/straddle gates enough to drive `div_fp → 0` also drives
`div_tp → 0` (byte-for-byte champion). There is **no operating point that recovers even
one division TP without an edge-FP regression**, so the `0.1 · division_jaccard` term
stays fully forgone. This confirms the family-mix / detection wall extends to
image-intensity-conditioned division recovery — the champion's dropped heads that survive
the geometric gate are not, in fact, true mitotic daughters, and no local intensity/shape
signature rescues them.

## Disposition

Champion pointer **unchanged** (no promotion): `champion/config.json` never enables
`division_overlay`. The `split-signature` detector stays in `src/biohub_tracking/`
default-off (like the prior rejected `nearest-head` / `mutual-nn` overlays) as an
available, byte-safe lever; the linking pipeline is byte-for-byte unchanged. **No Kaggle
submission** (CV-only child). Evidence appended to `docs/ai/experiment_ledger.jsonl`.
