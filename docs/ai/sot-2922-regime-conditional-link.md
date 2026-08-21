# SOT-2922 — Regime-conditional linking operating point (REJECTED)

**Cycle 5, child 2** (linking-side reformulation of the family-mix / density-mix
wall; independent of the detection-side [SOT-2923](sot-2923-regime-conditional-detect.md)
if present, and its twin diagnostic [SOT-2921](../ai/experiment_ledger.jsonl) density covariate).

## Hypothesis

Use SOT-2921's GT-free observable per-sequence density covariate (`median_knn_um`)
to **conditionally select the linking operating point per sequence**: apply the
globally-rejected aggressive levers — `motion_gain=2.0` (SOT-2900 reserve) and the
mutual-NN cycle-consistency prune (`cycle_consistency_gate`, SOT-2910) — in the
**dense** regime, and the champion (`motion_gain=1.0`, no prune) in the **sparse**
regime. The threshold τ is fit **leave-one-family-out (3-fit / 1-test)**, a
monotone rule in covariate space (never referencing family id). Goal: clear the
mandatory **4/4 per-dataset non-regression** gate that no single global config
passes, plus beat champion micro_adj **0.6760**.

## Result — REJECTED (4/4 non-regression fails)

Same-seed leak-free 4-family LOFO A/B, champion byte-frozen (sha256
`f2b1076…6522fc`, micro_adj 0.6760), detection cached once/family. Full evidence:
`experiments/sot2922/ab_regime_conditional_link.json`.

| metric | champion | conditional (LOFO) | Δ |
| --- | --- | --- | --- |
| micro_adj | 0.6760 | 0.6821 | **+0.0061** |
| micro_raw | 0.6960 | 0.7023 | +0.0063 |
| macro_adj | 0.7310 | 0.7575 | +0.0265 |
| lineage_macro_adj | 0.7351 | 0.7623 | +0.0272 |

Per-dataset Δ adjusted edge Jaccard: `44b6_0113de3b +0.0757`, `44b6_0b24845f
+0.0217`, `6bba_05db0fb1 +0.0091`, **`6bba_05b6850b −0.0009`** ⇒ **3/4**, gate
fails (`no_regression_adj = no_regression_raw = False`). micro/macro/lineage all
rising together is NOT sufficient — the strict per-dataset gate catches the one
held-out regression.

## Root cause — density-mix wall, linking side

Per-family adjusted-optimal linking op:

| family | median_knn_um (observable) | adj-optimal op | champion adj |
| --- | --- | --- | --- |
| 44b6_0113de3b | 8.54 | motion_gain=2.0, gate **off** | 0.9078 |
| 44b6_0b24845f | 8.45 | motion_gain=2.0, gate **off** | 0.6938 |
| 6bba_05db0fb1 | 7.45 (densest) | motion_gain=2.0, gate **off** | 0.7477 |
| **6bba_05b6850b** | **9.49 (sparsest)** | **motion_gain=1.0, gate ON (prune)** | 0.5748 |

Three of four families simply want plain `motion_gain=2.0`. The **only** family
that wants the mutual-NN prune is `6bba_05b6850b` — and it is the **observably
sparsest** sequence (largest `median_knn_um`). The prune sheds dense-volume FP
steals, yet the family that needs it looks *sparsest* by the observable covariate:
the covariate is **anti-aligned** with where the lever helps. A covariate-keyed
policy therefore assigns `6bba_05b6850b` to the champion (sparse) side and can
never select its prune. The LOFO fit collapses to `motion_gain=2.0` **everywhere**
(all four held-out ops = gain2/gate-off; none fell back to champion), adding
nothing beyond the already-known global gain=2 reserve and still failing on the
`6bba_05b6850b −0.0009` tick.

In-sample oracle upper bound (leaky — GT picks each family's op) = **0.6938**:
headroom exists, but it is unreachable leak-free through the observable density
covariate.

## Conclusion

This **re-confirms on the linking axis** what SOT-2923 confirmed on detection: the
observable density covariate crosscuts families and cannot select a per-regime
operating point that generalizes LOFO to the held-out family. The cycle-5
observable-covariate conditioning approach is non-promotable on both detection and
linking. The flag `regime_conditional_link` stays **default-off** (champion
byte-identical), the champion pointer is unchanged, and **no Kaggle submission**
was made. Next axis needs a stronger regime signal than observable density, or a
different lever entirely (parent SOT-2919 resume decision).
