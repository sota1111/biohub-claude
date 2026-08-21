# SOT-2910 — Bidirectional mutual-NN cycle-consistency edge gate (cycle 3)

**Axis.** Port the cycle-consistency principle (NeighborTrack arXiv:2211.06663,
DistNet2D arXiv:2310.19641): run the champion Hungarian NN linker and keep only the
`t → t+1` links that are a **mutual nearest neighbour** in both directions (forward
`argmin_j dist(i,j)` and backward `argmin_i dist(i,j)` agree), pruning the contested,
globally-assigned-but-not-mutual edges most likely to be an FP steal in dense volumes.
This **prunes** primary edges and adds none — a pure precision-for-recall trade on the
metric-highest-leverage term (`TP/(TP+FP+FN)`, FP edges hurt most; a shed FP edge also
frees the matched GT node after the `min_track_length=4` prune).

Default-off knobs `LinkParams.cycle_consistency_gate` (bool) + `cycle_consistency_margin`
(scaled µm, ambiguity margin). Off ⇒ champion **byte-for-byte**.

**Distinct from prior rejected axes** (not a retry): SOT-2895 suspicious-review is a
jump/reversal *self-motion* signature (non-specific); SOT-2898 `mutual-nn` was a
*division overlay that ADDED* second-daughter fork edges; SOT-2883 is a forward↔backward
*motion-field residual* gate reusing the SOT-2864 smoothed field. This axis is a pure
point mutual-NN **prune** needing no motion model.

## Result — REJECTED (same-seed A/B, family-mix wall)

`experiments/sot2910/ab_cycle_consistency.py` → `ab_cycle_consistency.json`. SOT-2903
re-anchored 4-family leak-free CV, detection cached once per family (only the linker
knob varies), champion byte-frozen (config.json sha256 `42064648…e01bdd`, verified
in-run). primary=`micro_adj`, guardrail=`micro_raw`. **No Kaggle submission.**

Margin sweep (champion micro_adj **0.6649**, micro_raw 0.6840, macro_adj 0.7180,
lineage_macro 0.7216):

| margin | micro_adj | Δadj | micro_raw | Δraw | per-ds adj no-reg | per-ds raw no-reg |
|--:|--:|--:|--:|--:|:--:|:--:|
| **0.0** | 0.6758 | **+0.0109** | 0.6911 | +0.0071 | **False** | **False** |
| 0.5 | 0.6726 | +0.0077 | 0.6857 | +0.0016 | False | False |
| 1.0 | 0.6633 | −0.0016 | 0.6735 | −0.0105 | False | False |
| 1.5 | 0.6506 | −0.0143 | 0.6577 | −0.0263 | False | False |
| 2.0 | 0.6446 | −0.0203 | 0.6486 | −0.0355 | False | False |

**Root cause = the same family-mix wall that sank SOT-2870/2883/2899.** At the best
point (margin 0.0) the micro *rises* +0.0109, but the gain is entirely the dense 6bba
lineage (95.8% of the micro weight): 6bba_05b6850b adj 0.5700→0.5930 (edge FP 215→183),
6bba_05db0fb1 0.7310→0.7350 (FP 148→112) — the mutual-NN prune genuinely sheds dense-volume
FP steals. But the clean sparse **44b6 lineage REGRESSES**: 44b6_0113de3b 0.8895→0.8734,
44b6_0b24845f 0.6817→0.6414 (−0.040). In a sparse family a cell's true successor is often
*not* its mutual nearest neighbour (Hungarian sacrifices local mutuality for a globally
cheaper valid assignment), so the gate cuts real TP edges there.

The tell that the micro gain is family-mix noise, not a real improvement: **macro_adj
0.7180→0.7107 and lineage_macro 0.7216→0.7142 both FALL** at margin 0 while micro rises —
the two family-mix-robust views move opposite the micro. No sweep point clears the
per-dataset non-regression gate on either the adjusted primary or the raw guardrail;
increasing the margin monotonically over-prunes recall and worsens every statistic.

**Disposition.** Knob stays default-off (champion config.json unchanged, byte-frozen).
No candidate artifact (non-promoted). pytest 248 green (incl. 6 new
`tests/test_cycle_consistency.py`), compileall green.
