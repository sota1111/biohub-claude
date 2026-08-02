# SOT-2305 — Re-anchor the local oracle to a multi-dataset GT holdout

Kaggle rank-improvement cycle 4 (parent SOT-2300), oracle-drift diagnosis axis.
**No Kaggle submission was made; no champion state was mutated.**

## Motivation — the oracle drift

Cycle 3 (SOT-2272) built a *local* promotion oracle from the only two ground-truth
families we hold, **both from the `44b6` dataset**: `44b6_0113de3b` (clean) and
`44b6_0b24845f` (fragmented/dim). On that 2-family oracle the DoG-v2 champion
measured micro-adj **0.7721**, a big jump over the v1 global-threshold baseline
(**0.5063**). The 2-family v1 score (0.5063) had reproduced the public LB (0.509),
so the oracle looked trustworthy — and DoG-v2 was promoted to champion.

The LB disagreed: v1 scored **0.509**, DoG-v2 scored **0.500**. The oracle said
"+0.266"; the LB said "−0.009". That divergence is **oracle drift**: the 2-family
`44b6`-only oracle over-weighted the one fragmented family DoG was tuned to recover
and was **blind to the two `6bba` families** that make up half of the real
four-dataset test set.

## What this cycle did

Re-anchored the local oracle onto **all four** test families the competition
actually scores — `44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`,
`6bba_05db0fb1` — by pairing each test `*.zarr` image with its train-split `*.geff`
ground truth (the same pairing that reproduced the LB for the two `44b6` families),
and produced a **per-dataset / per-prefix breakdown** for both detectors.

- Holdout data (`6bba` train GT + test images) was fetched into `data/` (gitignored,
  not redistributable). All four train GTs load with valid node counts
  (25755 / 32795 / 6362 / 69800).
- Script: `scripts/reanchor_oracle.py` (deterministic; re-running reproduces every
  score). Output: `docs/reanchored-oracle-evaluation.json`.

## Results — per-dataset breakdown

Adjusted edge Jaccard per family (`edge TP/FP/FN`, `pred_nodes` vs `n_true`):

| family | prefix | v1 global-thr | DoG-v2 |
| --- | --- | --- | --- |
| `44b6_0113de3b` (clean) | 44b6 | 0.9512 (47/2/3, 12269) | 0.8865 (47/2/3, 30691) |
| `44b6_0b24845f` (dim) | 44b6 | 0.0436 (2/1/47, 3599) | 0.6620 (38/5/11, 52213) |
| `6bba_05b6850b` (sparse) | 6bba | 0.7148 (597/29/248, 3405) | 0.2622 (619/251/226, 40450) |
| `6bba_05db0fb1` (dense) | 6bba | 0.0908 (101/26/1082, 8867) | 0.7141 (971/168/212, 74282) |

Micro-averaged (edge-count weighted, adjusted edge Jaccard):

| oracle | v1 global-thr | DoG-v2 | public LB |
| --- | --- | --- | --- |
| 2-family `44b6` (old oracle) | 0.5063 | **0.7721** | — |
| `6bba` only | 0.3526 | 0.5117 | — |
| **4-family holdout (re-anchored)** | **0.3598** | **0.5225** | v1 0.509 / DoG 0.500 |

## LB-divergence quantification (the headline)

- **The 2-family oracle was hiding DoG's degradation.** DoG-v2's micro collapses
  from **0.7721 → 0.5225** once the two `6bba` families are added — i.e. it moves
  right onto the DoG public LB of **0.500**. The inflated +0.266 the 2-family oracle
  reported was an artifact of scoring only the family DoG was tuned to recover. This
  is exactly the "hidden 6bba behaviour" the re-anchoring set out to expose. ✅
- **But the re-anchored oracle is not yet fully LB-faithful.** It still ranks
  DoG-v2 (0.5225) **well above** v1 (0.3598), whereas the LB ranks them ~tied with v1
  *marginally higher* (0.509 vs 0.500). And v1's holdout micro (0.3598) does **not**
  reproduce its own LB (0.509) the way the 2-family score (0.5063) coincidentally did.
- **Why:** the two detectors have **complementary per-dataset failure modes** that
  micro-averaging does not reconcile the way the LB does. v1 under-detects the dense
  `6bba_05db0fb1` (only 101 of ~1183 GT edges; adj 0.091). DoG over-detects the sparse
  `6bba_05b6850b` (pred 40450 vs 6362 true nodes → the node-count penalty crushes adj
  to 0.262). Whichever family dominates the (edge-count) weighting swings the micro,
  so the absolute values and the ranking still diverge from the LB. This points at a
  residual GT-pairing / normalization mismatch on the `6bba` split (the train `*.geff`
  may not be the exact hidden test GT for those families), which no local re-scoring
  can close on its own.

## Conclusion / handoff

The 4-family holdout **strictly supersedes** the 2-family `44b6` oracle as the screen
basis — it no longer hides DoG's `6bba` behaviour and it correctly deflates DoG's
score onto the LB. But it is **not yet a trustworthy LB proxy for champion
selection**, because it still mis-ranks v1 vs DoG relative to the LB. Recorded as
**`inconclusive`** in `docs/ai/experiment_ledger.jsonl`.

Follow-up children (both blocked on this one):

- **SOT-2306** — re-judge the detection champion on the re-anchored holdout
  (DoG vs global-threshold; revert if DoG is a net regression on the representative
  set). The per-dataset table above already shows DoG is a **regression on
  `6bba_05b6850b`** (0.7148 → 0.2622) even while it wins the dim `44b6` family, which
  is the crux of that decision.
- **SOT-2307** — make detection robust across datasets with adaptive intensity
  normalization, to fix the complementary v1/DoG per-family failures (dense
  under-detection vs sparse over-detection) instead of trading one for the other.

## Reproduce

```bash
.venv/bin/python scripts/reanchor_oracle.py            # all 4 families, both detectors
.venv/bin/python scripts/reanchor_oracle.py --families 44b6   # legacy 2-family oracle only
```

Deterministic — no RNG in the pipeline; every score re-runs identically. Full numbers
in `docs/reanchored-oracle-evaluation.json`.
