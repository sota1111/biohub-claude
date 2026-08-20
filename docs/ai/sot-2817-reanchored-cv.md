# SOT-2817 — Re-anchor the leak-free CV to the full competition metric + representativeness guard

**Type:** evaluation-system re-anchoring (oracle改定). **No champion change, no Kaggle submission.**
**Blocked-by:** SOT-2816 (oracle audit). **Parent:** SOT-2815 (cycle-2, 2nd series).

## Why (from the SOT-2816 audit)

The promotion oracle (leak-free CV) rose monotonically v1→v4 (0.3598 → 0.5225 → 0.6232 → **0.6649**)
while the public LB *fell* (0.624 → 0.509). SOT-2816 attributed that drop to a **competition
metric-patch re-score of a byte-frozen artifact — not a champion regression** (the champion has been
byte-identical since the commit that produced 0.624; the same deterministic submission was re-scored
to 0.509 after a global leaderboard re-score, topScore 0.943 → 0.957). The audit also measured two
concrete representativeness limits of the CV:

1. **The `0.1 · division_jaccard` term was reported as `null`** whenever the confusion matrix was
   empty, so the full competition metric `adjusted_edge_jaccard + 0.1 · division_jaccard` was not
   reported *consistently*, and a division-forfeiting config's 0.1 contribution silently vanished
   instead of being an explicit `0`.
2. **The micro is 95.8% 6bba-weighted** (sample weight `w = tp+fp+fn`: 44b6 = 4.2%, 6bba = 95.8%),
   so a change that helps the dense 6bba lineage a hair while hurting the sparse 44b6 lineage can
   raise the micro while lowering the LB. The micro is a *screen*, not a faithful LB ranker.

Because the audit concluded **metric-patch (not regression)** and did **not** surface a config that
beats the champion on the re-anchored CV, this issue re-anchors **the evaluation system only** and
leaves the champion **byte-invariant** (per the issue's explicit branch: "同一artifactの再採点で
LB低下 ⇒ champion は据置、再アンカリングは評価系のみ").

## What changed (`src/biohub_tracking/eval/cv.py`)

### 1. Full competition metric, always, with an explicit division term
`aggregate()` now always reports the complete metric with an **explicit** `division_term`:

- `division_measurable = (div_tp + div_fp + div_fn) > 0`. When a GT division event exists or a fork
  is predicted, the division Jaccard is a real number (`0.0` when the champion forfeits divisions —
  `allow_division=false` ⇒ `tp=fp=0, fn>0` ⇒ Jaccard `0.0`), and `division_term = 0.1 · Jaccard`.
- When the confusion matrix is empty (no GT division event *and* no predicted fork), the Jaccard is
  undefined (`NaN`/`null`) but `division_term = 0.0` **explicitly** — the 0.1 contribution is never
  dropped. `score = micro_adj + division_term` on every run.

This is byte-neutral for the champion: `division_term = 0.0`, so `score == micro_adj == 0.6649`.

### 2. Representativeness / robustness guard (micro **and** macro)
`CvResult` now carries family-mix-robust views alongside the sample-weighted micro:

| field | meaning |
| --- | --- |
| `macro_adj_edge_jaccard` | unweighted mean over families — each embryo video counts equally, so a dense family's raw edge count can't dominate |
| `lineage_macro_adj` | mean over the two **lineages** of each lineage's weighted micro — the sparse 44b6 lineage contributes at **parity** with dense 6bba |
| `by_lineage_weight_share` | the share of the micro weight each lineage carries (the 95.8% 6bba domination is *reported*, not hidden) |

`representativeness_report()` (pure function) self-checks whether the micro can be trusted as an LB
ranker:

- `dominant_lineage` / `dominant_lineage_weight_share` — which lineage carries the micro and by how
  much.
- `micro_lineage_macro_gap` / `family_mix_sensitive` — the gap between the sample-weighted micro and
  the parity-weighted lineage macro; over `MIX_SENSITIVITY_TOL = 0.05` the CV **ranking** is flagged
  family-mix sensitive, so a bare micro gain is **not sufficient** evidence for promotion (gate on
  the macro / per-family no-regression too).
- `cv_public_order_consistent` / `cv_public_same_magnitude` — the micro is not below and is within
  one order of magnitude (`MAGNITUDE_RATIO_BOUND = 2.0`) of the public LB best; a CV a decimal order
  off the LB is not measuring the same thing (the "桁一致" sanity セルフチェック).

On the current champion the guard fires as expected: micro 0.6649 vs lineage-macro **0.7216**
(gap 0.0567 > 0.05) ⇒ `family_mix_sensitive = True`, `dominant_lineage = 6bba` (share 0.9575). The CV
correctly warns that its own ranking leans on the dense lineage.

### 3. Regression guard = the CV arithmetic *is* `eval/score.py`
`test_cv_arithmetic_matches_score_py` re-scores the same `(pred, gt)` pairs + `n_true` through both
`aggregate()` and `eval.score.evaluate_datasets` and asserts `micro_edge_jaccard`,
`micro_adj_edge_jaccard`, `division_jaccard`, and `score` all agree to `rel_tol=1e-12` — the oracle
cannot drift from the scorer. `--check-champion` now guards **both** the micro-adj (0.6649) **and**
the full re-anchored score (0.6649), pinning the division-term wiring.

## Leak-free invariants (maintained — unchanged by this issue)

The re-anchoring is a *reporting/representativeness* change; it does **not** touch the leak-free
holdout design (see `docs/ai/sot-2761-leak-free-cv.md`), which remains in force:

- **Entity holdout** — the scored unit is a whole embryo video (`.geff` lineage), never a frame/node;
  the four families are the exact Kaggle test set (two `44b6` + two `6bba`), so no cell/track
  straddles the train↔score boundary.
- **Temporal / causal holdout** — detection is per-timepoint and linking is forward-only
  (`t → t+1`); no future frame informs an earlier prediction, nothing is fit across time.
- **Selection discipline** — with four videos, the micro alone overfits; callers gate on per-family
  `no_regression_vs` **and** (now) the family-mix-robust macro when `family_mix_sensitive` is set.

## Champion re-decision — **byte-invariant** (evidence)

The audit ruled out regression and surfaced no config that beats the champion on the re-anchored CV,
so the champion pointer is **not** changed:

- `git diff` on `champion/config.json`, `registry.json`, and `submit/kernel/biohub-claude-champion.py`
  vs `main` is **empty** (verified in the PR).
- Re-scoring every historical config through the re-anchored `aggregate()`
  (`experiments/sot2817-reanchor/rescore_reanchored.py`, pure — no data) reproduces the registry
  numbers exactly and leaves v4-shorttrack the top config (re-anchored score **0.6649**), so no
  re-anchored evidence supports a pointer move.

Promotion is decided on the **re-anchored CV**, never on public raw values (no public-best chasing).

## Verification evidence — re-anchored CV order vs real LB

`experiments/sot2817-reanchor/reanchored_cv_rescore.json` (re-aggregated from the SOT-2816 per-family
counts). Re-anchored score vs the two **independently submitted** public-LB points:

| Config | Re-anchored CV | Public LB | division term |
| --- | ---: | ---: | ---: |
| detect-link-v1 | 0.3598 | 0.509 | 0.0 (explicit) |
| detect-link-dog-v2 | 0.5225 | — | 0.0 (explicit) |
| detect-link-dog-v3-adaptive | 0.6232 | — | 0.0 (explicit) |
| **detect-link-dog-v4-shorttrack** | **0.6649** | 0.624 | 0.0 (explicit) |

On the two submitted points the CV order **matches** the LB order (v1 < v4 on both) —
`cv_lb_order_consistent_on_submitted = true`, i.e. the re-anchored CV is **not** inversely correlated
with the LB (the acceptance bar: "少なくとも逆相関でない"). Caveat: only two of the four configs were
independently submitted, so this is a 2-point order check, not a rank correlation; the remaining
representativeness risk (the +0.041 CV→hidden optimism gap and the 6bba domination) is exactly what
`family_mix_sensitive` now flags at report time. The division term is `0.1 · 0 = 0` for every config
(all forfeit divisions) and is now reported explicitly rather than as `null` — recovering it is
SOT-2818's job, screened on **this** re-anchored CV with a representative division sample.

## Pre-submission checklist (oracle side — integrates the guard)

Before treating a CV number as promotion evidence:

1. **Full metric reported** — `score = micro_adj + division_term`, `division_term` explicit (never
   `null`); if `division_measurable` is false the term is a real `0.0`, not a dropped field.
2. **Representativeness read** — check `family_mix_sensitive`. If set, a micro gain is insufficient:
   require per-family `no_regression_vs` **and** a non-regressing `lineage_macro_adj` / `macro`.
3. **CV↔LB sanity** — `cv_public_order_consistent` and `cv_public_same_magnitude` both true; a CV
   below or an order off the public best is a red flag to investigate before promoting.
4. **Arithmetic pinned** — `python -m biohub_tracking.eval.cv --check-champion` reproduces
   micro-adj **and** full score 0.6649; `pytest -q` green (incl. the score.py-parity regression test).
5. **No public-best chasing** — the promotion decision cites the re-anchored CV, never the raw public
   number.
