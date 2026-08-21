# SOT-2883 — Ultrack bidirectional motion-consistency link gate (cycle-8, REJECTED)

**Axis.** Port Ultrack's (Nature Methods 2025, royerlab) *temporal-consistency selection =
adjacent-frame overlap maximization* to the point-detection **linking** stage as a
bidirectional forward↔backward agreement gate, layered on the confirmed-robust SOT-2864
motion-model linker (`motion_model_link` + `motion_gate_on_prediction`, micro-adj **0.6760**,
the first & only linking lever to improve all four LOFO families with zero per-dataset
regression).

**Mechanism (`link_consistency_gate`, default-off).** For a link `t→t+1`, `i→j`:
- **forward residual** `r_f = ||src_pred_fwd[i] − dst[j]||` — the SOT-2864 global smoothed
  motion field predicts where `src[i]` moves.
- **backward residual** `r_b = ||dst_pred_bwd[j] − src[i]||` — the SAME field estimated on the
  *reversed* frame pair predicts where `dst[j]` came from.

A genuine cell has both small; an FP-prone link (the forward field over-smoothing onto a
spurious near neighbour) has a large residual in one direction the backward field disagrees
with. The gate **penalises** (`link_consistency_weight * 0.5*(r_f+r_b)` added to the LAP cost)
and/or **rejects** (`link_consistency_tol` hard-drops a pair with `r_f>tol` OR `r_b>tol`). It is
a pure restriction of the SOT-2864 feasible set, so `max_distance` is preserved by construction.

**Distinct from prior rejected linking axes.** Not SOT-2871's carried single-direction
running-average velocity (family-mix sensitive); not SOT-2870's learned one-direction p_edge —
this is a *symmetric motion-field cross-check* (an overlap surrogate).

## Result — REJECTED / non-promotable

Same-seed leak-free 4-family LOFO A/B on the champion classical detection (cached once per
family). Champion reproduced byte-exact (micro-adj **0.6649**); SOT-2864 motion reference
reproduced **0.6760** (+0.0111, all-4-family non-regressing). Screen
`experiments/sot2883/screen_consistency_gate.json`.

| variant | micro-adj | ΔvsChamp | ΔvsMotionRef | no-reg vs champ | no-reg vs ref | family-mix-sensitive |
|---|---|---|---|---|---|---|
| motion ref (SOT-2864) | 0.6760 | +0.0111 | — | ✓ | — | yes (gap 0.059) |
| hard tol=6.0 | 0.6798 | +0.0149 | +0.0038 | ✓ | ✗ | yes |
| hard tol=4.0 | 0.6672 | +0.0023 | −0.0088 | ✗ | ✗ | no |
| hard tol=2.5 | 0.5975 | −0.0674 | −0.0785 | ✗ | ✗ | no |
| hard tol=1.5 | 0.3808 | −0.2841 | −0.2952 | ✗ | ✗ | yes |
| soft weight=1.0 | 0.6772 | +0.0123 | **+0.0012** | ✓ | **✓** | **yes (gap 0.059)** |
| soft weight=0.25/0.5 | 0.6760 | +0.0111 | −0.000 | ✓ | ✓ | yes |

**Why non-promotion:**

1. **Only `soft weight=1.0` clears per-dataset non-regression vs BOTH baselines**, but its
   gain over the incumbent SOT-2864 motion reference is **+0.0012** — inside the noise band and
   **entirely from the dominant 6bba lineage**: both sparse 44b6 families are *byte-identical* to
   the motion ref (0.9078 / 0.6938), the whole delta is 6bba_05db0fb1 +0.0019 & 6bba_05b6850b
   +0.0005. It is **family_mix_sensitive=True** (micro↔lineage-macro gap 0.0586 > 0.05), and the
   mix-robust views are flat (lineage-macro 0.7351→0.7358, macro 0.7310→0.7316).
2. **Doctrine:** a family-mix-sensitive micro gain is not sufficient promotion evidence
   (SOT-2830 rejected at +0.0022 for exactly this; SOT-2816 documented CV +0.042 → public −0.115).
   A +0.0012 6bba-only delta is far below the transfer-confidence bar to displace the robust,
   all-4-family SOT-2864 lever.
3. **The stronger-micro hard-gate variant (tol=6.0, +0.0149)** regresses BOTH sparse 44b6
   families vs the motion reference (44b6_0113de3b 0.9078→0.8897, 44b6_0b24845f 0.6938→0.6901;
   `beats_ref=False`) — it trades the robust lever for a lineage-skewed one. Reject.
4. The mechanism *does* remove a few real FP edges on the dense families (6bba_05db0fb1 dFP −20 at
   weight=1.0) but recovers nothing beyond the forward-only motion field on the sparse lineage, so
   it yields no robust cross-family improvement over the incumbent.

**Ship state:** `link_consistency_gate` retained **default-off** (champion config carries no
`link_consistency*` key; champion CV reproduces 0.6649 byte-exact; `tests/test_consistency_gate.py`
7 green, full suite 218 green, exec-compat OK). **No Kaggle submission** (parent resume responsibility).
SOT-2864 motion linking (+0.0111, all-4-family non-regressing) remains the #1 reserve/final-window
LB-probe candidate — this axis did not beat it robustly.
