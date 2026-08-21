# SOT-2929 — Finer-grain leak-free CV holdout + CV↔public transfer-trust

**Direction B of SOT-2927 cycle-6** (escalation ladder step 2–3 = oracle re-anchor /
generalization-gap diagnosis). This child repairs the **evaluation base itself**, not the
champion. **No Kaggle submission; champion byte-frozen.**

Module: `src/biohub_tracking/eval/holdout.py` · tests: `tests/test_holdout.py` ·
artifact: `experiments/sot2929/transfer_trust.json` (pure, data-free — reuses the historical
per-family counts baked into `eval/transfer.py`).

## Problem

The primary leak-free CV (`eval/cv.py`) splits its holdout by **embryo lineage**
(`44b6` vs `6bba`) — a leave-one-family-out (LOFO) grouping. SOT-2921 proved that grouping
**crosscuts the true difficulty boundary**: the `6bba` lineage contains *both* the sparsest
sequence (`6bba_05b6850b`, observed `median_knn_um` ≈ 9.49 µm) *and* the densest
(`6bba_05db0fb1` ≈ 7.45 µm). Every regime-conditioned promotion (SOT-2922 linking / SOT-2923
detection / SOT-2931 soft mixture) then failed the 4/4 per-dataset gate on exactly the family
whose lineage label disagreed with its density regime. The question this child answers: **is a
holdout finer than the lineage LOFO a better private-LB proxy — or a worse one?**

## Finer-grain leak-free holdout (implemented)

Two leak-free refinements over the lineage LOFO, each a **strict partition of the four CV
families** (cover + disjoint), so no whole embryo video is ever split across buckets (entity
holdout preserved — the scored unit is a whole `.geff` lineage):

1. **per-sequence parity** — each embryo video is its own singleton holdout unit and counts
   equally (the finest leak-free unit).
2. **observable density-regime stratification** — sequences are bucketed `sparse`/`dense` by the
   **GT-free** `median_knn_um` covariate (`eval/regime.py`, SOT-2921), computable from the test
   detection point cloud alone. The threshold (`derive_regime_threshold` = midpoint of the largest
   covariate gap above the value median = **9.017 µm**) isolates the sparse tail from the
   covariate distribution alone — **no family prefix, no ground truth**.

**Leak-free audit** (`leak_free_audit`, pinned by tests):

| property | value |
| --- | --- |
| holdout unit | whole embryo video (`.geff` lineage) |
| strict partition of `CV_HOLDOUT` | ✅ |
| regime derived from observable covariate (`assign_regime`) | ✅ |
| **regime crosscuts lineage** | ✅ `6bba → {dense, sparse}`, `44b6 → {dense}` |

The regime split `sparse={6bba_05b6850b}` / `dense={44b6_0113de3b, 44b6_0b24845f, 6bba_05db0fb1}`
**respects the family-internal heterogeneity the lineage LOFO cannot express** — the two `6bba`
sequences land in different regimes.

## CV↔public transfer-trust table

Each candidate holdout statistic scored on the historical champion lineage; Spearman ρ measured
against the **same-metric** public anchors (v1 0.509 / v2 0.500 / v4-champion 0.624; v3-adaptive
`public_lb=None`, excluded — unconfirmed cross-patch footing).

| config | public | micro_adj | lineage_macro | per_seq_parity | regime_parity | **sparse** | dense |
| --- | --- | --- | --- | --- | --- | --- | --- |
| detect-link-v1 | 0.509 | 0.3598 | 0.4295 | 0.4501 | 0.4190 | **0.7148** | 0.1231 |
| detect-link-dog-v2 | 0.500 | 0.5225 | 0.6419 | 0.6312 | 0.4903 | 0.2622 | 0.7183 |
| dog-v3-adaptive | *(none)* | 0.6232 | 0.6942 | 0.6898 | 0.6083 | 0.5025 | 0.7141 |
| dog-v4-shorttrack (champ) | 0.624 | 0.6649 | 0.7216 | 0.7180 | 0.6524 | 0.5700 | 0.7349 |

**Transfer-trust (Spearman vs public, 3 anchors):**

| statistic | ρ vs public | crowns the public winner (champion)? |
| --- | --- | --- |
| micro_adj (live KPI) | +0.50 | ✅ |
| lineage_macro_adj (LOFO parity) | +0.50 | ✅ |
| per_sequence_parity | +0.50 | ✅ |
| regime_parity_adj | +0.50 | ✅ |
| **sparse_regime_adj** | +0.50 | ❌ **crowns v1, not the champion** |
| dense_regime_adj | +0.50 | ✅ |

(The +0.50 ceiling is the v1↔v2 0.509/0.500 near-tie; at the 4-anchor resolution with v3=0.557
the blended statistics all rise to ρ=+0.80 and the two single-regime statistics fall to +0.40 —
consistent with the SOT-2903 adjusted-vs-penalty-free audit.)

## Conclusion — CV is a **sufficient** proxy at the blended granularity, **insufficient** at single-regime granularity

- Every **parity/blended** statistic (lineage-parity, per-sequence-parity, regime-parity) ranks
  the anchored public configs **identically to the live KPI** and crowns the champion. Making the
  holdout finer this way is leak-free but adds **no discriminating power** — granularity is not
  the binding constraint.
- The **single sparse-regime** statistic is a **strictly weaker** private proxy: on the sparsest
  sequence (`6bba_05b6850b`) the champion is *not* the best config (v1's `0.7148` > champion's
  `0.5700`), so a sparse-only holdout **inverts the public crown**. This is the same density-mix
  crosscut that broke SOT-2922/2923/2931 — now shown to also be a *scoring* hazard, not only a
  *tuning* one: over-weighting one difficulty stratum optimizes the wrong config.
- Therefore the finer holdout is adopted only as a **per-regime no-regression guardrail** (which
  `cv.py`'s per-family `no_regression_vs` already enforces), **not** as a replacement promotion
  KPI. `RECOMMENDED_KPI` stays `micro_adj` (the official full metric).
- **Binding oracle limitation = public-anchor scarcity** (3 confirmed same-metric points, champion
  dominates every blended statistic), *not* holdout crosscut. Send-up to the parent (SOT-2927):
  escalate to acquiring more independent same-metric public anchors / the private-anchored
  two-signal gate, not to further CV-granularity redesign.

**Ledger:** `result=rejected` for the hypothesis "a finer-grain regime-stratified holdout is a
better/necessary private proxy than the lineage-parity adjusted micro" (cycle 6). Champion
pointer unchanged; no submission.
