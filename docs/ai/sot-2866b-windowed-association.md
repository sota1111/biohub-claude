# SOT-2871 — Portable Trackastra-style windowed global association (+ parental-softmax)

Cycle-7 linking axis under parent **SOT-2866** (biohub-claude Kaggle 順位向上サイクル第7次).

**Verdict: REJECTED — kept default-OFF, champion byte-frozen, no Kaggle submission.**

## Hypothesis

Extend the champion's memoryless `t -> t+1` greedy bipartite match into a portable
Trackastra (arXiv:2405.15700) **short sliding-window** global association (W = 2–3
frames), so a cross-hop **carried velocity** couples adjacent transitions and the
window can recover 1-frame-gap FN edges that greedy matching mislinks. Torch /
attention / pretrained-weights free — **numpy/scipy only** (a LAP chain with
birth/death outlier arcs; SOT-2864 motion field as the base cost; optional
parental-softmax division add-on).

Distinct from the prior rejected linking axes (not a blind retry):

- **SOT-2763 static gap-closing** — REJECTED; the official edge metric drops the
  non-consecutive bridge edge. *This emits only consecutive `t->t+1` edges.*
- **SOT-2849 node-interp gap-recovery** — REJECTED; family-mix-sensitive node
  insertion. *This inserts no nodes; it only changes the primary link cost.*
- **SOT-2830 / SOT-2840 global min-cost-flow** — +0.0022, family-mix-sensitive; its
  pure-distance cost **decoupled** per transition. *Here a carried window velocity
  couples adjacent transitions so the window is meant to bite.*
- **SOT-2864 motion field** — +0.0111 pure motion field, **no cross-hop carry**.
  This A/B's whole question: *does the windowed carry beat pure motion?*

## Implementation (default-off infrastructure)

`src/biohub_tracking/link.py`:
- `LinkParams.window_assoc` (default **1** = untouched champion path, byte-for-byte),
  `window_theta`, `window_carry_weight`, `window_parental_softmax`,
  `window_softmax_min_share`, `window_softmax_temp`.
- `_window_assign` — one motion-predicted LAP transition with birth/death outlier
  acceptance (`theta`); a strict generalisation of the champion `_assign` (with
  `theta=inf`, `src_pred=None`, no penalty terms it is byte-identical).
- `_window_link` — time-ordered `t->t+1` chain; each source's predicted displacement
  blends the SOT-2864 global motion field with a carried velocity running-averaged
  over the previous `window_assoc-1` transitions (reset across any frame gap).
- `_parental_softmax_divide` — attaches a **balanced** second daughter only when its
  softmax association share (over the parent's feasible children within
  `division_distance`) clears `window_softmax_min_share`; out-degree capped at 2.
- `champion_params` passes the new keys through; absent keys ⇒ `window_assoc=1` ⇒
  champion reproduced byte-for-byte.

`src/biohub_tracking/champion.py`: threads the six knobs (all default-off).

Tests: `tests/test_window_link.py` (7) — off-by-default == champion; motion-off +
`theta=inf` reduces to champion assignment; birth/death threshold suppresses a
marginal link; carried velocity bites vs the memoryless champion; only consecutive
edges, history resets across a gap; parental-softmax attaches a balanced daughter
but rejects a distant decoy; determinism.

## Leak-free CV A/B (SOT-2761 4-family LOFO, same-seed)

`experiments/sot2866b/screen_windowed_association.py` →
`screen_windowed_association.json`. Baseline champion `micro_adj_edge_jaccard =
0.6649` (reproduced exactly ⇒ byte-invariance confirmed). Reference: SOT-2864
motion-only (already merged, default-off) = **0.6760** (+0.0111).

| variant | score | Δ vs champion | edge Δ (tp/fp/fn) | per-dataset no-reg | family-mix-sensitive |
| --- | --- | --- | --- | --- | --- |
| champion baseline | 0.6649 | — | — | — | — |
| **ref: motion-only (SOT-2864)** | **0.6760** | **+0.0111** | +16 / −20 / −16 | yes | yes |
| W=3 carry=0.5 gate=True | 0.6732 | +0.0083 | +8 / −20 / −8 | **yes** | yes |
| W=3 carry=0.5 gate=False | 0.6665 | +0.0016 | −2 / −9 / +2 | no | yes |
| W=2 carry=0.5 gate=False | 0.6613 | −0.0036 | −8 / +2 / +8 | no | yes |
| W=2 carry=0.5 gate=True | 0.6603 | −0.0046 | −14 / −3 / +14 | no | yes |
| W=3 carry=1.0 gate=False | 0.6601 | −0.0048 | −10 / +4 / +10 | no | yes |
| W=2 carry=1.0 gate=False | 0.6529 | −0.0120 | −19 / +18 / +19 | no | yes |
| W=3 carry=1.0 gate=True | 0.6536 | −0.0113 | −33 / +0 / +33 | no | yes |
| W=2 carry=1.0 gate=True | 0.6383 | −0.0266 | −68 / +9 / +68 | no | yes |
| W=2 parental-softmax (allow_division) | 0.6354 | −0.0295 | −51 / +40 / +51 | no | yes |

`n_promotable_vs_champion = 1`, **`n_beats_motion_reference = 0`**.

## Why rejected

1. **The window does not beat pure motion.** The single per-dataset-non-regressing
   windowed variant (W=3, carry=0.5, gate=True, 0.6732) is **below** the already-
   merged SOT-2864 motion-only reference (0.6760). The cross-hop carried velocity —
   the whole mechanistic novelty vs SOT-2830 — subtracts value rather than adding it;
   at `carry=1.0` (trust the trajectory only) every window regresses hard. The window
   does not "bite" beyond what the SOT-2864 motion field already delivers.
2. **Family-mix-sensitive.** Every variant is `family_mix_sensitive=True`; the 6bba
   lineage carries 95.75% of the CV weight, so any apparent gain is not a robust
   cross-family signal (the same fragility that sank SOT-2830/2840/2849).
3. **Parental-softmax hurts.** The division add-on is the worst variant (−0.0295,
   +40 FP): sparse-GT over-count penalty punishes the extra daughters — consistent
   with the cycle-3/4 detection-stage finding that no single global operating point
   exists on this metric.

Escalation-ladder note (design §41-51): this axis *was* the external-knowledge /
architecture-change rung (a ported top-solution mechanism, not more local tuning).
It fails to beat the existing motion lever, reinforcing that the linking stage is
saturated on this leak-free CV. Infra is retained default-off so a future cycle can
reuse the windowed LAP scaffold if a genuinely better edge cost appears; champion
stays byte-frozen and nothing is submitted.
