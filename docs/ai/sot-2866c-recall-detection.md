# SOT-2866c — Recall-oriented FN-edge-endpoint recovery (leak-free CV A/B)

**Cycle 7, child 3** on the biohub-claude Kaggle rank-improvement ladder (parent
SOT-2866, external-knowledge port). Champion `detect-link-dog-v4-shorttrack`
`config.json` sha256 `42064648e612183e761bf9d40b70d3e8a2497453a878f1a44f5b52e410e01bdd`
**byte-frozen** (leak-free CV micro-adj **0.6649**). No champion / registry
mutation. **No Kaggle submission** (per issue).

## Hypothesis (external-knowledge / metric-property exploit)

The competition edge metric
([royerlab metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md))
scores an edge TP only when **both** endpoints match a GT node within **7 µm**,
but charges **no node FP** for a predicted node that matches nothing (the GT is
sparse) — only the mild *global* over-prediction penalty
`J·(1 − 0.1·(N_pred − N_true)/N_true)`. So a predicted node added **below** the
champion's strict adaptive cutoff can only *help* (recover a missed-detection
FN-edge endpoint → +TP, −FN) or be *free* (unmatched ⇒ no node FP), paying only
the shared over-prediction penalty. The axis isolates **GT-node recall @7 µm** as
the objective (not the aggregate score, not a family-invariant per-voxel operating
point) and measures the **recall-vs-over-prediction-penalty tradeoff** directly.

## Implementation (default-off, non-destructive)

- `detect.py`: `DetectParams.recall_recovery = ("madk_tier", k_low, max_extra_frac)`
  (default `None` → champion byte-for-byte). It keeps the champion primary peaks
  (`response > threshold`) **unchanged** and additionally admits the strongest
  local maxima in the band `median + k_low·1.4826·MAD < response <= threshold`
  (`k_low` **below** the champion `mad_k = 3.0`, so the tier is strictly the peaks
  the strict cutoff dropped), capped to `floor(max_extra_frac · n_primary)`. Every
  added peak is dimmer than every primary peak, so the brightest-first ordering is
  preserved and the primary tier is byte-frozen. NMS scalar-gate (champion) path
  only; requires `mad_k`; incompatible with the `local_threshold` surface.
- `eval/recall_metric.py`: `gt_node_recall` — the **GT-node recall @7 µm**
  evaluator over the *same* optimal-bipartite 7 µm matching the edge metric uses.
  Reports overall node recall and **edge-endpoint** node recall (denominator =
  GT nodes incident to a GT edge — the only ones that can become an edge TP).
- Wired through `champion.champion_params` (absent config key ⇒ off).
- Tests: `tests/test_recall_recovery.py`, `tests/test_recall_metric.py`.

## Evidence — SOT-2761 leak-free CV, champion frozen, single-variable same-seed A/B

Detection re-runs the **real** `recall_recovery` code path per variant; the
champion linker re-links; scoring goes through the one SOT-2761 CV aggregation
(champion reproduced 0.6649 byte-for-byte). Grid `k_low ∈ {2.0, 1.0}` ×
`max_extra_frac ∈ {0.1, 0.3, 0.6}`. `experiments/sot2866c/screen_recall.json`.

**Baseline champion** — GT-node edge-endpoint recall @7 µm is already near-ceiling:

| family | adj | pred_nodes | ep-recall@7µm |
|---|---|---|---|
| 44b6_0113de3b | 0.8895 | 29844 | **1.0000** (no headroom) |
| 44b6_0b24845f | 0.6817 | 34485 | 0.8824 |
| 6bba_05b6850b | 0.5700 | 10938 | 0.9373 |
| 6bba_05db0fb1 | 0.7310 | 69784 | 0.9162 |
| **micro** | **0.6649** | | **0.9257** |

**Recall-vs-penalty tradeoff (micro):**

| k_low | frac | micro-adj | Δscore | ep-recall | Δep-recall | ΔTP | ΔFP | ΔFN | ΔN_pred | no-reg |
|---|---|---|---|---|---|---|---|---|---|---|
| — champion | 0.6649 | — | 0.9257 | — | — | — | — | — | — |
| 2.0 | 0.1 | **0.6692** | **+0.0043** | 0.9439 | +0.0182 | +33 | +8 | −33 | +11097 | ✗ |
| 1.0 | 0.1 | 0.6670 | +0.0021 | 0.9435 | +0.0178 | +31 | +9 | −31 | +13413 | ✗ |
| 2.0 | 0.3 | 0.6601 | −0.0048 | 0.9439 | +0.0182 | +32 | +10 | −32 | +20075 | ✗ |
| 1.0 | 0.3 | 0.6555 | −0.0094 | 0.9439 | +0.0182 | +30 | +16 | −30 | +26675 | ✗ |
| 2.0 | 0.6 | 0.6513 | −0.0136 | 0.9439 | +0.0182 | +32 | +10 | −32 | +25201 | ✗ |
| 1.0 | 0.6 | 0.6430 | −0.0219 | 0.9439 | +0.0182 | +30 | +16 | −30 | +42528 | ✗ |

**The recall gain saturates immediately** (ep-recall +0.0182 at frac=0.1 already;
no further recall at frac 0.3/0.6) while the **over-prediction penalty grows
monotonically** with the added-node count — so the micro peaks at the *smallest*
tier (+0.0043) and goes net-negative thereafter. Classic recall–penalty curve:
recall a step then flat; penalty linear; net = quick peak then decline.

## Decision — REJECTED (non-promotion), champion byte-frozen

**No variant passes the mandatory per-dataset no-regression gate** (0/6). The
per-family edge deltas expose *why*, and it is a sharper finding than "the micro
is family-mix-sensitive":

- **The recovered FN edges are confined to ONE family, `6bba_05db0fb1`**
  (ΔTP +30…+32, ΔFN −30…−32). `44b6_0b24845f` recovers a single edge (+1 TP).
- **`6bba_05b6850b`: ΔTP = ΔFP = ΔFN = 0 at every setting** — up to +5695 added
  detections recover **zero** edges. Its missed endpoints (ep-recall 0.9373) are
  **not detection-limited**: a detection already sits there (lost at the
  matching/linking stage, or the miss is outside the sparse-annotated region), so
  more detections only absorb the global penalty (adj 0.5700 → 0.5150 @ f0.6).
- **`44b6_0113de3b`: ΔTP = 0 always** — already ep-recall 1.0, zero headroom, so
  it can only *lose* to the penalty (0.8895 → 0.8354 @ f0.6).

Because two of four families have **no recall headroom** yet every family pays the
**global** over-prediction penalty, a *uniform* recall tier is per-dataset
regressing by construction. The best micro (+0.0043 @ k2.0/f0.1) is smaller than
SOT-2849's +0.0124 and, like it, family-mix-sensitive.

## New grounds vs SOT-2789 (operating-point枯渇) — an evidence-backed advance, not a blind retry

- **SOT-2789 / SOT-2848,2863** searched for a *family-invariant per-voxel
  threshold/magnitude* on the **aggregate score** and found none.
- **This issue** set **GT-node recall @7 µm** as the objective under the metric's
  **no-node-FP** property and **did move recall (+0.0182)** — proving detection
  headroom *exists* and the no-node-FP asymmetry is real. The **new** finding is
  that recovery is **detection-limited in only one of four families**: adding a
  global sub-threshold tier cannot target `6bba_05db0fb1`'s specific missed
  endpoints without paying the over-prediction penalty on the two families whose
  misses are not detection-limited. The champion's own operating point was never
  moved (primary tier byte-frozen); the lever fails at the **selection gate**, not
  the recall objective.

The default-off `recall_recovery` + `gt_node_recall` infra is retained
(byte-frozen champion) for a future *targeted* (per-family / motion-gated)
recovery that could avoid the uniform-penalty failure mode. No Kaggle submission;
champion sha256 `42064648…` unchanged.
