# SOT-2930 — Cycle-6 direction A': wholesale public classical baseline foundation

Kaggle 順位向上サイクル第2次 (parent **SOT-2927**), explore direction **A'** — adopt an
**independently-constructed public classical baseline FOUNDATION wholesale** (not a
single-lever cherry-pick), measure it on the leak-free 4-family LOFO CV as an independent
candidate, keep the champion as hedge. **No Kaggle submission; champion byte-frozen.**

## Why this axis (role A')

Our champion `detect-link-dog-v4-shorttrack-motion-gain1` (leak-free CV micro_adj
**0.6760**, public **0.626**) was built by *incremental single-lever accretion* on a
classical DoG-detect / nearest-neighbour-link substrate and is stuck at a local optimum on
the density-mix wall — SOT-2864 (ARGUS global motion model) was the last promotion, then
SOT-2895/2898/2899/2910/2911/2918/2920/2922/2923/2931 were all rejected/inconclusive. The
hypothesis: an **independently-constructed** classical foundation may sit in a *different
basin* and either (a) beat the champion on leak-free CV → two-signal-gate promotion, or
(b) provide a structurally-independent hedge for final-slot diversification.

## What was adopted (source + portability)

| lineage | public LB | portable part (adopted) |
| --- | --- | --- |
| [`xiaoleilian/biohub-cell-tracking-classical-baseline`](https://www.kaggle.com/code/xiaoleilian/biohub-cell-tracking-classical-baseline) | ~0.720 | DoG detect + **memoryless two-pass tight(7µm)→full(11µm) Hungarian** link ("v4: tight gate first, then full gate for leftovers") + division |
| [`kaiwalyaatulraut/biohub-cell-tracking-solution`](https://www.kaggle.com/code/kaiwalyaatulraut/biohub-cell-tracking-solution) | — | corroborating classical DoG + greedy-NN family |

- **PORTABLE / adopted wholesale**: the baseline's *structural linking foundation* — a
  memoryless two-pass tight-then-full Hungarian assignment with its division-on /
  short-track-prune envelope. This is the baseline's basin, **structurally distinct** from
  our champion's global-motion-model LAP linker (the champion's `motion_model_link` is
  turned OFF here — this is the baseline's own operating point, not a champion micro-diff).
- **SHARED SUBSTRATE**: DoG blob detection. Our champion detector *is itself* the ported
  classical-baseline-family DoG (`detect-link-dog-v4`); the baseline's differentiator vs
  our champion is the *linker*, not the detector, so detection is held at the shared
  classical operating point (which also caches detection once per family → the run is a
  clean linking-only ablation).
- **Reconstruction disclosure**: the private notebook's exact detection thresholds are not
  byte-recoverable offline. This is a faithful reconstruction from the documented design +
  the repo's already-ported baseline linker knobs (`link_two_pass` / `link_full_distance`,
  SOT-2899), **not** a byte-copy of the private kernel source. Disclosed.
- **NON-PORTABLE / excluded**: none for the *classical* baseline (no GPU weights). The
  frontier 0.89+ learned UNet+ILP lineages are out of scope (offline / no-weights).

## Leak-free CV A/B (same-seed 4-family LOFO)

Harness: `experiments/sot2930/ab_public_classical_foundation.py` → `evaluate_cv(config)`
(the SOT-2761 leak-free CV). Champion reproduced its registry reference **0.6760**
exactly. Three faithful wholesale variants, all sharing the champion DoG detection, LINK
block replaced wholesale by the baseline foundation:

| variant | linking foundation | micro_adj | Δ vs champ | 4/4 non-reg | div_jac |
| --- | --- | --- | --- | --- | --- |
| **champion** | ARGUS global motion model LAP | **0.6760** | — | — | 0.0 |
| w1 two-pass core | memoryless two-pass, div off, prune 4 | 0.6588 | **−0.0172** | ❌ | 0.0 |
| w2 two-pass + division | + `allow_division`, div_dist 11 | 0.5998 | −0.0762 | ❌ | 0.01 |
| w3 + no pruning | + `min_track_length=1` | 0.5940 | −0.0820 | ❌ | 0.0101 |

Per-dataset Δadj of the best variant (w1): `44b6_0113de3b −0.0350`, `44b6_0b24845f +0.0003`,
`6bba_05b6850b −0.0076`, `6bba_05db0fb1 −0.0270`. The champion's smoothed motion field
wins precisely on the clean/dense families (`44b6_0113de3b`, `6bba_05db0fb1`) where a
memoryless assignment steals distractors. Adding division makes it far worse: the sparse-GT
over-prediction penalty punishes the extra fork edges while `division_jaccard` stays ~0.01.

## Verdict — REJECTED (not promoted, not hedged); champion byte-frozen

- **Two-signal gate did NOT fire.** Every wholesale variant is CV-inferior; none clears
  the mandatory *CV-up + 4/4 per-dataset non-regression* gate. → **not promoted.**
- **rogii discipline.** The adopted baseline's public 0.720 **exceeds** our champion's
  0.626, yet on our **leak-free CV** the memoryless two-pass foundation *underperforms* the
  accreted motion-model champion. High public of an adopted baseline is **not** private/CV
  evidence — this is exactly the rogii trap, and the gate correctly refuses it.
- **No hedge either.** Because the foundation is CV-inferior by a clear margin (−0.0172 at
  best), spending a scarce final submission slot to hedge on it would be a public-chasing
  bet the rogii post-mortem forbids. → **not adopted as a hedge.** (It may be reconsidered
  as diversification only if a future cycle proves the CV is non-representative; it is not
  now.)
- Champion stays the live config, **byte-frozen** (`sha256 15e1db9c…`). **No Kaggle
  submission** (per this child's scope).

## Reproduce

```bash
cd /workspaces/ai-dev-control-plane/.targets/biohub-claude
.venv/bin/python -m experiments.sot2930.ab_public_classical_foundation
# -> experiments/sot2930/ab_public_classical_foundation.json
#    VERDICT any_promote=False best=w1_twopass_core best_micro_adj=0.6588 champion=0.6760
```

Deterministic (no RNG). Candidate config +full variant record:
`champion/candidates/sot2930-public-classical-foundation.json`.
