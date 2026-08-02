# SOT-2306 — Re-decide the detection champion on the re-anchored oracle

**Cycle 4 · biohub-claude · axis: DoG family vs pre-DoG global-threshold on the LB-representative holdout**

## Question

When SOT-2272 replaced the v1 global-intensity-threshold detector with DoG-v2, the
public LB moved the *wrong* way (0.509 → 0.500). SOT-2305 then showed the old
2-family (44b6-only) oracle had hidden DoG's over-detection on the 6bba half of the
real test set. SOT-2306 asks: on the SOT-2305 4-dataset LB-representative holdout,
A/B the DoG detector against the global-threshold baseline and **lock in the
LB-consistent champion — reverting to global-threshold if DoG degrades**.

Since filing, the sibling SOT-2307 promoted a third detector,
`detect-link-dog-v3-adaptive` (DoG + a robust per-volume `median + 3·1.4826·MAD`
threshold that prunes the spurious detections the node-count penalty punishes). That
is exactly the "intermediate DoG + FP-suppression" variant SOT-2306 lists as option
(c), so the A/B is run **three-way**.

## A/B result — 4-family holdout, size-weighted adjusted edge Jaccard

The metric is the size-weighted **adjusted** edge Jaccard
(`edge_jaccard · min(1, n_true/n_pred)`) — the node-count over-detection penalty is
part of the real Kaggle metric. Holdout = the exact four Kaggle test families
(`test/*.zarr` image + `train/*.geff` GT).

| Detector | 44b6 | 6bba | **holdout micro-adj** | public LB |
| --- | --- | --- | --- | --- |
| v1_global_threshold (pre-DoG) | 0.5063 | 0.3526 | **0.3598** | 0.509 |
| dog_v2 (fixed percentile-92) | 0.7721 | 0.5117 | **0.5225** | 0.500 |
| **v3_adaptive (current champion)** | 0.7716 | 0.6168 | **0.6232** | not submitted |

Ranking: **v3_adaptive (0.6232) > dog_v2 (0.5225) ≫ v1_global_threshold (0.3598)**.

Deterministic `run_pipeline` (no RNG); these numbers reproduce the SOT-2305 /
SOT-2307 measurements exactly. Full data: `experiments/sot2306/champion_redecide.json`.

## Decision — KEEP champion (revert REJECTED)

The DoG family **strictly dominates** global-threshold on the re-anchored oracle, so
the LB-suggested "DoG is worse, revert to global-threshold" hypothesis is **rejected**.
The champion is **maintained** at `detect-link-dog-v3-adaptive`.

- `champion/config.json` is **unchanged** (already v3-adaptive, the A/B winner);
  `EMBEDDED_CHAMPION_CONFIG` stays byte-for-byte in sync — `tests/test_exec_compat.py`
  and the full 55-test suite pass.
- Why v1 collapses on the holdout: it under-detects the dense `6bba_05db0fb1`
  (adj 0.091) and the fragmented `44b6_0b24845f` (adj 0.044). DoG+adaptive keeps all
  four families balanced (44b6 0.7716, 6bba 0.6168) with per-dataset precision/recall
  in the 0.71–0.96 / 0.73–0.94 range — no dataset collapses.

## Known limitation (inherited from SOT-2305)

The re-anchored oracle is **not yet a faithful LB proxy for the v1-vs-DoG ranking**:
v1's holdout micro-adj (~0.36) does not reproduce its own LB (0.509), and v1/dog_v2
are near-tied on the LB (0.509 vs 0.500) whereas the oracle ranks DoG far above v1.
This residual GT/normalization gap cannot be closed without a Kaggle submission, which
SOT-2306 forbids. The decision therefore uses the **best available** oracle; on it the
DoG family clearly wins, so no revert is warranted. Closing the residual oracle↔LB gap
(e.g. blind LB probing of v3-adaptive, or further GT normalization) remains future work.

## No Kaggle submission

Per the issue, this is a read-only A/B + champion decision. **No submission was made.**
