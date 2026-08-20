# SOT-2762 — Division-aware linking + division Jaccard (REJECTED, champion maintained)

**Cycle 2 (biohub-claude Kaggle 順位向上第2次).** Axis: enable division-aware
linking as a score lever and measure division Jaccard on the SOT-2761 leak-free
CV. **Verdict: rejected** — the champion `detect-link-dog-v4-shorttrack`
(`allow_division=False`) is maintained unchanged. No Kaggle submission.

## Motivation

The official metric is `adjusted_edge_jaccard + 0.1 · division_jaccard`
(royerlab). The champion runs `link.allow_division=False`, so it never predicts a
fork and its division Jaccard is **0.0**: the four-family holdout has 3 GT
divisions (all in `6bba_05db0fb1`) and the champion misses every one (division
`tp=0 / fp=0 / fn=3`). Zebrafish embryogenesis is division-heavy, so on paper the
division term is an untapped +0.1-weighted score source.

## What was implemented

Division linking already existed (`link.allow_division` / `division_distance`);
the gap was **over-split control**, because indiscriminate division sprays
spurious forks on dense volumes. Added to `LinkParams` (`src/biohub_tracking/link.py`):

- **`division_max_sibling_ratio`** (default `0.0` = off) — a sibling-balance gate.
  A leftover `t+1` detection is attached as a parent's second daughter only if its
  scaled parent-distance `d2 ≤ ratio · d1`, where `d1` is the parent→primary
  (one-to-one assigned) daughter distance. A real mitotic split yields two
  daughters roughly symmetric about the parent, so a detection much farther than
  the assigned child is not a plausible sibling. Plumbed through `champion_params`.
- **`division_distance`** (existing) reused as the second over-split lever: tighter
  than `max_distance` restricts divisions to close, high-confidence pairs.

Unit tests: `tests/test_link.py::test_sibling_ratio_gate_{suppresses_unbalanced,keeps_balanced}_division`.

## Screen (single-variable ablation, deterministic ⇒ same-seed A/B)

`experiments/sot2762/screen_division.py` — detection frozen (DoG-v3 `mad_k=3.0`)
and computed once per family; every link variant re-linked off cached detections
and scored through the one SOT-2761 CV aggregation (`eval.cv`). Grid:
`division_distance ∈ {7,5,3,2}` × `division_max_sibling_ratio ∈ {0,3,2,1.5}`.

| variant | micro_adj | division_J | score | Δ vs champ | no-regression |
|---|---|---|---|---|---|
| **champion (division off)** | **0.6649** | **0.0** | **0.6649** | — | — |
| dd=5.0, ratio=1.5 (best composite / max div_J) | 0.6584 | 0.0625 | 0.6646 | −0.0003 | ✗ |
| dd=5.0, ratio=2.0 | 0.6581 | 0.0435 | 0.6625 | −0.0024 | ✗ |
| dd=3.0, ratio=2.0 | 0.6632 | 0.0 | 0.6632 | −0.0017 | ✗ |
| dd=2.0, any ratio | 0.6649 | 0.0 | 0.6649 | −0.0000 | ✗ |
| dd=7.0, ratio=0.0 (unconstrained) | 0.6296 | 0.0172 | 0.6313 | −0.0336 | ✗ |

**Every** division variant regresses. `n_promotable = 0`.

## Why it fails (mechanism)

At the best variant (dd=5/ratio=1.5) division recovers **1 of 3** GT divisions
(`6bba_05db0fb1` `tp=1/fp=3/fn=2`) while spraying **10 FP forks** on the dense
`6bba_05b6850b` family. The added fork edges regress edge adj on 3/4 families
(`6bba_05b6850b` 0.5700→0.5616, `6bba_05db0fb1` 0.7310→0.7261, `44b6_0113de3b`
0.8895→0.8887). The 0.1 division weight buys at most +0.0063 here, which cannot
pay for the edge-Jaccard cost of the fork edges on the dense families. Tightening
`division_distance` to 2 µm makes division essentially never fire (div_J 0.0,
score == champion) — there are no close, free divisions to recover. The over-split
knobs behave exactly as designed (tighter dd + balance ratio raise div_J per fork
and cut edge damage) but the trade never nets positive **and** never clears the
per-dataset no-regression gate.

## Confirm

`experiments/sot2762/confirm_division.py` re-scores the champion baseline and the
best challenger through the real `champion_params(config) → run_pipeline` +
`evaluate_cv` (re-runs detection, not cached) — the same independent re-score that
caught the SOT-2369 plumbing bug. It asserts `division_max_sibling_ratio` survives
`champion_params`, and reproduces the screen: champion 0.6649 (div_J 0.0),
challenger dd5/ratio1.5 0.6646 (div_J 0.0625, Δ −0.0003, no-regression ✗),
`promotable=False`.

## Decision

Champion **maintained** at `detect-link-dog-v4-shorttrack`. `champion/config.json`,
`registry.json`, and `EMBEDDED_CHAMPION_CONFIG` are unchanged (still
`allow_division=False`; exec-compat gate + pytest green). The division over-split
mechanism (`division_max_sibling_ratio`, default off) is kept as a documented
default-off knob, exactly as the SOT-2369 motion lever (`velocity_gain`) was kept
after its own rejection — future cycles can revisit division only if a detector or
matcher change first makes divisions recoverable without dense-family fork FP.
Rejected axis recorded in `docs/ai/experiment_ledger.jsonl`. No Kaggle submission
(the parent SOT-2757 owns submission).
