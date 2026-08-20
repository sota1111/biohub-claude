# SOT-2816 — Oracle audit: CV↔LB divergence attribution + division CV headroom

**Type:** diagnostic / oracle audit (no champion change, no submission, no promotion).
**Question:** the primary KPI (leak-free CV) *rose* across cycles (0.6232 → 0.6649) while the
public LB *fell* (0.624 → 0.509). Is that drop **metric-patch re-scoring** or **champion
regression**, and can the current CV be trusted as the promotion oracle?

**Reproduce:** `experiments/sot2816-oracle-audit/rescore_lineage.py` (lineage re-score + division
headroom) and `python -m biohub_tracking.eval.cv --check-champion` (champion regression guard). Raw
outputs: `experiments/sot2816-oracle-audit/lineage_cv_rescore.json`,
`experiments/sot2816-oracle-audit/champion_cv_repro.json`.

---

## 1. Historical champion × CV series (one leak-free harness) vs real LB points

Every historical champion config was re-scored through the **single** leak-free CV evaluator
`biohub_tracking.eval.cv.evaluate_cv` (the same 4-family holdout, same micro-averaging), so the CV
column is produced by one oracle, not four re-implementations. Configs reconstructed from the
champion docs/registry are in `experiments/sot2816-oracle-audit/config_*.json` (they differ from the
frozen champion only by the documented knobs: DoG on/off, `mad_k`, `min_track_length`).

| Champion (chrono) | Issue | Leak-free CV micro-adj | 44b6 | 6bba | Submitted → public LB |
| --- | --- | ---: | ---: | ---: | --- |
| detect-link-v1 (global-intensity) | SOT-1983 | **0.3598** | 0.5063 | 0.3526 | 2026-07-28 → **0.509** |
| detect-link-dog-v2 (DoG p92) | SOT-2272 | **0.5225** | 0.7722 | 0.5117 | (near-tied v1 on LB ~0.50) |
| detect-link-dog-v3-adaptive (MAD) | SOT-2307 | **0.6232** | 0.7716 | 0.6168 | 2026-08-02 submitted (ref 55193790, artifact `01c2f393…`, PENDING) |
| **detect-link-dog-v4-shorttrack** | SOT-2369 | **0.6649** | 0.7836 | 0.6596 | 2026-08-03 → **0.624**; 2026-08-20 15:22 (same frozen artifact) → **0.509** |

Notes:
- The re-score **reproduces the registry values exactly** (v3 0.6232, v4 0.6649) and the ledger's
  earlier one-off numbers (v1 0.3598, dog-v2 0.5225) — the CV series is internally consistent and
  **monotonically rising v1→v4** on the micro. `--check-champion` reproduces **0.6649 == 0.6649
  (delta 0.0000)**.
- Per-family, the lineage is **not** monotone (a warning sign for a 4-video oracle): v1 already
  scored 6bba_05b6850b **0.7148**, which DoG-v2 *dropped* to 0.2622 (over-detection) before v3's MAD
  cutoff recovered it to 0.5025; v1 could not see the dim cell in 44b6_0b24845f (0.0436) that DoG
  fixed (0.662). Every promotion was gated on micro **and** per-family no-regression, but the micro
  is dominated by two families (see §4).

## 2. Attribution of 0.624 → 0.509

**Verdict: the drop is NOT a champion regression. It is best explained by a competition
re-score (metric-patch) of a byte-frozen artifact; the exact magnitude split is inconclusive.**

Evidence that **rules out champion regression** for this move:
- The champion has been **byte-frozen since the SOT-2369 commit (`57bdf4f`) that produced the
  0.624 submission on 2026-08-03**. `git diff 57bdf4f..HEAD -- champion/config.json
  submit/kernel/biohub-claude-champion.py registry.json` is **empty**.
- Fingerprints match the ledger across every intervening cycle-2/3/4 aggregation:
  `champion/config.json` sha256 **`42064648…`**, submit kernel sha256 **`48b1eaa2…`** (verified this
  run). Every cycle-2/3/4 child merged as a **default-off** knob that resolves to `None`, so the
  effective pipeline never changed.
- No worse config was ever promoted or submitted after v4. There is **no regressed artifact** that
  could have produced 0.509.

Evidence that the drop is a **scoring change on a fixed artifact** (metric-patch):
- The **same** frozen v4 artifact is behind both the 2026-08-03 = 0.624 and the 2026-08-20 15:22 =
  0.509 observations (the pipeline is deterministic; kernel byte-identical → same `submission.csv`).
  Same input, two scores ⇒ the *scorer* changed, not the submission.
- The whole leaderboard was re-scored in that window: `docs/ai/kaggle/leaderboard-rank.jsonl`
  records **topScore 0.943 (2026-08-02) → 0.957 (2026-08-20)** — a global re-score, consistent with
  the documented **division-jaccard exploit patch** that re-scored submissions
  (royerlab/kaggle-cell-tracking-competition; see §3 sources).
- Mechanistic tie-in: the champion runs `allow_division=false` and forfeits the entire `0.1 ·
  division_jaccard` term. A re-score that changed how the division term (or the exploit it patched)
  contributes to the hidden-set score moves a division-forfeiting submission's number without any
  code change on our side.

**Residual uncertainty (why the *magnitude* split is inconclusive, not the direction):**
- We cannot independently re-fetch the identical artifact's pre- vs post-patch pair of scores
  (Kaggle re-submission is forbidden by this issue), so we cannot put a number on "how many of the
  0.115 points are the metric re-score."
- The machine record (`leaderboard-rank.jsonl`) stores **cumulative-best** public score
  (0.509 @08-02, 0.624 @08-20 08:16), while the human notes log per-submission points
  (0.557/0.624/0.509). The `bestPublicScore` field cannot be decomposed into per-submission scores.
- A pre-existing **CV→hidden-LB optimism gap of +0.041** is already measured (CV 0.6649 vs public
  best 0.624; train-`.geff` GT vs hidden-scored subset). Part of any CV↔LB gap is this optimism, not
  the patch. Separating the two requires a fresh submission.

Conclusion: **direction = metric-patch (regression ruled out); magnitude decomposition =
inconclusive/diagnostic.** The rising CV is not contradicted by a regressed champion — the champion
never regressed — so the divergence is an **oracle-representativeness + scoring-change** phenomenon,
which is exactly what SOT-2817 (re-anchor the CV to the full metric + representativeness guard) is
scoped to fix.

## 3. Metric definition check + division-exploit patch provenance

The official score is **`adjusted_edge_jaccard + 0.1 · division_jaccard`** with
`adjusted_edge_jaccard = max(0, J · (1 − 0.1 · (N_pred − N_true)/N_true))`.

Repo implementation matches this exactly:
- `src/biohub_tracking/eval/score.py`: `ADJUSTMENT_ALPHA = 0.1`, `SCORE_DIVISION_WEIGHT = 0.1`,
  `adjusted_edge_jaccard()` = `max(0, J·(1 − 0.1·(N_pred−N_true)/N_true))`,
  `score = adj_edge_jaccard + 0.1·division_jaccard`.
- `src/biohub_tracking/eval/division_metric.py`: division Jaccard = `TP/(TP+FP+FN)` over the
  `grandparent→divider→children→grandchildren` window; clean-room port of the reference rules.

**Sources (division-jaccard exploit re-score):**
- https://github.com/royerlab/kaggle-cell-tracking-competition
- https://raw.githubusercontent.com/royerlab/kaggle-cell-tracking-competition/main/metrics.md
- Competition discussion: submissions were re-scored after a division-jaccard exploit patch
  (recorded in the SOT-2815 cycle-2b axis-selection ledger entry; corroborated here by the LB
  topScore shift 0.943→0.957).

## 4. Division term CV headroom — the 0.1 term is **not robustly measurable** on this CV

Measured this run (`load_geff` out-degree ≥ 2 per holdout family):

| Family | GT division events |
| --- | ---: |
| 44b6_0113de3b | 0 |
| 44b6_0b24845f | 0 |
| 6bba_05b6850b | 0 |
| 6bba_05db0fb1 | **3** |
| **Total** | **3** |

- **Champion contribution:** `allow_division=false` ⇒ 0 predicted forks ⇒
  `division_tp=0, fp=0, fn=3` ⇒ `division_jaccard = 0` ⇒ **+0.1·0 = 0** (confirmed empirically in
  `champion_cv_repro.json`: 6bba_05db0fb1 `division_fn=3`, overall `division_jaccard=0.0`).
- **Ceiling:** full recall, no FP → `3/3 = 1.0` → **+0.1** to score.
- **FP fragility (from full recall):** 1 spurious fork → `3/4 = 0.75` → **+0.075** (Δ −0.025);
  2 FP → 0.6 → +0.06; 3 FP → 0.5 → +0.05.
- **Recall increments (no FP):** recover 1 of 3 → `1/3` division Jaccard → **+0.0333** to score;
  2/3 → **+0.0667**; 3/3 → **+0.1** (the ceiling).

**"Not robustly measurable" verdict (numbers):**
1. **Single-video, 3-sample point statistic — zero cross-video replication.** All 3 GT division
   events are in **one** holdout video (6bba_05db0fb1); the other three videos have **0**. So the
   division Jaccard is a 3-positive point estimate on a single embryo with no per-video variance and
   no way to tell a generalizing division mechanism from one that overfits that one video. The CV's
   whole purpose — cross-family no-regression — is undefined for a term only three families can't
   even exercise.
2. **FP-dominated denominator.** You cannot earn the term without enabling fork prediction
   (`allow_division=true`), which on a 69,784-node prediction emits uncontrolled FP forks (exactly
   the SOT-2762 failure). With a denominator of only 3 true events, the division Jaccard is set by
   the FP count, not by the recall: each spurious fork costs −0.025, so a handful of FPs (routine on
   a 69k-node graph) sinks the term below its +0.0333-per-recovered-event signal. The metric is
   governed by noise it cannot control at 3-event resolution.

Together: recovering a true division is worth only +0.0333, it is observable on a single video, and
enabling the mechanism that earns it injects FP forks that dominate the 3-event denominator. This
quantifies why SOT-2762's division-linking work was rejected — the CV cannot robustly score the 0.1
term — and is why SOT-2818 must screen→confirm a division overlay on the **re-anchored** CV (with a
representative division sample), not this one.

## 5. Sparse-GT holdout LB representativeness — micro is 6bba-dominated

The reported micro-adjusted edge Jaccard is weight-averaged by sample size `w = TP+FP+FN`:

| Family | weight `w` | share of micro |
| --- | ---: | ---: |
| 44b6_0113de3b | 52 | 2.1% |
| 44b6_0b24845f | 54 | 2.2% |
| 6bba_05b6850b | 1060 | 42.4% |
| 6bba_05db0fb1 | 1331 | 53.3% |
| **Total** | **2497** | 100% |

- The two dense **6bba** families carry **95.8%** of the micro weight; the two sparse **44b6**
  families carry **4.2%**. The headline CV number is ~96% a 6bba statistic.
- Consequence for representativeness: if the hidden LB test set weights families differently (or the
  sparse 44b6 behaviour matters more on the hidden split), the CV micro **mis-ranks** — a change
  that helps 6bba by a hair and hurts 44b6 can raise the CV while lowering the LB. Combined with the
  per-family non-monotonicity in §1 and the +0.041 optimism gap, this is a concrete
  representativeness limit: **a 4-video, 6bba-dominated micro is a screen, not a faithful LB
  ranker.** (This is the hand-off to SOT-2817: add a representativeness guard, e.g. per-family /
  per-lineage reporting weighted toward the LB's family mix, not raw edge counts.)

## 6. Regression guard + byte-invariance (no champion change, no submission)

- `python -m biohub_tracking.eval.cv --check-champion` → **champion CV reproduced: 0.6649 == 0.6649
  (delta 0.0000)**.
- `git diff HEAD -- champion/config.json registry.json` (vs the pre-audit tree) is **empty** — this
  audit adds only `docs/ai/*` + `experiments/sot2816-oracle-audit/*` + a ledger entry.
- Fingerprints unchanged: `champion/config.json` sha256 `42064648…`, submit kernel sha256
  `48b1eaa2…`. **No Kaggle submission** (child; submission is the parent resume run's job).

## Bottom line

1. The CV is internally consistent and rose monotonically v1→v4; `--check-champion` reproduces
   0.6649 exactly.
2. **The 0.624→0.509 LB drop is not a champion regression** — the champion has been byte-frozen
   since the commit that produced 0.624, so the same artifact was re-scored to 0.509. The direction
   is **metric-patch / competition re-score** (corroborated by topScore 0.943→0.957); the exact
   metric-patch-vs-optimism magnitude split is **inconclusive** without a forbidden re-submission.
3. The `0.1·division_jaccard` term is **not robustly measurable** on this CV (3 events, SNR ≪ 1).
4. The CV micro is **95.8% 6bba-weighted**, a representativeness limit that (with the optimism gap)
   explains a rising CV alongside a re-scored LB. → hand off to SOT-2817 (re-anchor) / SOT-2818
   (division overlay on the re-anchored CV).
