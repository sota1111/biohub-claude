# SOT-2894 — Re-anchor leak-free CV to the CV↔public negative transfer

**Type:** diagnostic / CV re-anchor (no champion change, **no Kaggle submission**, no promotion).
**Gate:** no cycle-2 promotion decision until this re-anchor is recorded (issue mandate).
Champion `champion/config.json` sha256 `42064648…` — **byte-frozen** (verified this run).

**Reproduce:**
- `python -m biohub_tracking.eval.cv --check-champion` → champion CV reproduced **0.6649 == 0.6649 (delta 0.0000)** (live from `data/`).
- `python -m biohub_tracking.eval.transfer --out …` → the pure ranking table + order-consistency.
- `python experiments/sot2894-cv-reanchor/reanchor_transfer.py` → live champion guard + cross-check of the reconstructed lineage against the SOT-2816 single-harness re-score.
- `pytest tests/test_transfer.py` → the pure re-anchor logic.

---

## 0. TL;DR — the premise was partly mis-posed; the transfer failure is now localised

The issue frames the rate-limiter as a **CV↔public negative transfer**: the leak-free CV *rose*
(0.6232 → 0.6649) while public *fell* (0.624 → 0.509). Reconstructing all four historical champion
configs on the **one** leak-free harness and scoring them against the *same-metric* public anchors
shows:

1. **There is NO cross-config negative transfer.** On the same-metric (pre-patch) public anchors
   the CV is *positively* correlated with public (Spearman **+0.5** on every candidate statistic)
   and the champion `detect-link-dog-v4-shorttrack` is the **top** on both the CV *and* the public
   LB (0.624). The only discordance is the v1↔dog-v2 **near-tie** (public 0.509 vs 0.500 — within LB
   noise).
2. **The apparent "CV up / public down" is one config across a metric boundary.** Public 0.624
   (2026-08-03) and 0.509 (2026-08-20) are the **byte-identical** frozen champion re-scored by the
   competition's **division-exploit metric patch** (leaderboard topScore 0.943 → 0.957 in the same
   window) — SOT-2816's verdict, reproduced here. `0.624 ≡ 0.509` is **not two orderable configs**,
   so no config-scoring CV can "place 0.509 below 0.624" — that criterion is **inconclusive by
   construction** and recorded as such.
3. **The one genuine internal CV defect — the node-count-penalty tail — is now isolated and
   removed.** The entire dog-v2 → v3 CV "gain" (+0.10 on the legacy `micro_adj`) was **node-count-
   penalty relief**, not better matching. The re-anchored primary KPI (`lineage_macro_raw`) strips
   that layer so future promotions cannot bank it.

---

## 1. Ranking table — 4 historical configs, one leak-free CV (criterion 1)

Every config re-scored through the single `biohub_tracking.eval.cv` harness (same 4-family holdout,
same micro-averaging). `micro_adj` is the legacy primary KPI; `micro_raw` is the penalty-stripped
matching quality (micro edge Jaccard); `lineage_macro_raw` is the **re-anchored transfer-robust KPI**
(raw edge Jaccard, lineage parity); `node-penalty` = `micro_adj − micro_raw`.

| config | issue | micro_adj (legacy) | micro_raw | lineage_macro_raw (re-anchor) | node-penalty | public LB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| detect-link-v1 | SOT-1983 | 0.3598 | 0.3419 | 0.4077 | +0.0179 | **0.509** (07-28, pre-patch) |
| detect-link-dog-v2 | SOT-2272 | 0.5225 | 0.6561 | 0.7258 | −0.1336 | **~0.500** (pre-patch, near-tie) |
| detect-link-dog-v3-adaptive | SOT-2307 | 0.6232 | 0.6531 | 0.7198 | −0.0299 | — (ref 55193790 pending; 0.557 candidate, **unconfirmed**) |
| **detect-link-dog-v4-shorttrack** | SOT-2369 | **0.6649** | 0.6840 | **0.7358** | −0.0191 | **0.624** (08-03 pre-patch) → 0.509 (08-20 **same artifact**, post-patch) |

The champion `micro_adj` reproduces the registry **0.6649** byte-for-byte (live `--check-champion`).

## 2. Negative-transfer layer attribution (criterion 2)

Same-harness A/B (only the config differs — the definition of a controlled ablation):

- **Node-count-penalty tail (identified, removed).** dog-v2 → v3 raises `micro_adj` +0.10
  (0.5225 → 0.6232) but `micro_raw` **falls** (0.6561 → 0.6531) and `lineage_macro_raw` **falls**
  (0.7258 → 0.7198). The gain is entirely the node-count penalty moving from **−0.1336 to −0.0299**
  (the MAD threshold cut 6bba_05b6850b over-detection 40450 → 13376 pred nodes) — the registry
  reason for v3 already admits "the 6bba_05b6850b matched-edge TP/FP/FN is unchanged (619/251/226) —
  the whole gain is pruning ~27000 spurious detections the penalty punished." `N_true` is a coarse
  `estimated_number_of_nodes` proxy; a CV that can be climbed +0.10 by tuning predicted-node **count**
  against that proxy — with matching quality flat — is exactly a non-transferring lever. **Removed**
  by making `lineage_macro_raw` (penalty-free) the primary KPI.
- **Family-mix / lineage weighting (already guarded, SOT-2817).** The `micro_adj` is 95.8%
  6bba-weighted; the re-anchor uses **lineage parity** so the sparse 44b6 lineage is not drowned.
- **External division-exploit metric-patch (dominant, external).** The 0.624 → 0.509 move is a
  re-score of the byte-frozen champion, corroborated by topScore 0.943 → 0.957 and the champion's
  `allow_division=false` forfeit of the patched 0.1·division term (SOT-2816 §2-3). This layer lives
  in the **hidden scorer**, not our config, so no local CV can neutralise it — it can only be
  *disclosed* (done) and not chased.

**Inconclusive (recorded, per the issue's escape clause):** the exact magnitude split between the
metric-patch and the pre-existing CV→hidden **optimism gap** (+0.041: CV 0.6649 vs public-best
0.624) cannot be measured without a forbidden re-submission. The v3-adaptive → public mapping (the
human-noted 0.557) is likewise **unconfirmed** (ref 55193790 was pending), so v3 is excluded from the
order-consistency anchor rather than mis-anchored.

## 3. Re-anchored CV & the "champion 0.509 lowest" criterion (criterion 3)

- **Achievable part — done.** The re-anchored KPI `lineage_macro_raw` (a) removes the node-count-
  penalty illusion so a promotion cannot bank over-detection relief, and (b) enforces lineage parity.
  On the same-metric public anchors it stays **order-consistent** (Spearman +0.5, champion CV-top ==
  public-top), so the CV crowns the config the pre-patch LB crowns.
- **Unachievable-by-construction part — recorded inconclusive.** "Place the champion's 0.509 below
  0.624" is ill-posed: those two public points are the **same** byte-frozen config re-scored across
  the division-exploit patch. No config-scoring CV can order one config below itself. Placing the
  champion "lowest" would also be **wrong**: on same-metric footing the champion is the public **best**
  (0.624 > v1 0.509 > dog-v2 0.500) and the local best-matching config, so demoting it would
  re-introduce the very oracle drift this issue guards against. The honest re-anchor therefore keeps
  the champion top **and** exposes the +0.041 optimism gap + the external metric-patch as the residual
  CV↔public deltas — neither of which is a CV mis-ranking.

## 4. Byte-invariance & no submission (criterion 4)

- `python -m biohub_tracking.eval.cv --check-champion` → **0.6649 == 0.6649 (delta 0.0000)**.
- `git diff HEAD -- champion/config.json registry.json submit/kernel/` is **empty**; champion
  sha256 `42064648…` unchanged. This work adds only `src/biohub_tracking/eval/transfer.py`,
  `tests/test_transfer.py`, `experiments/sot2894-cv-reanchor/*`, `docs/ai/*`, and a ledger entry.
- **No Kaggle submission** (submission is the parent resume run's job; champion frozen).

## Bottom line

The cycle-2 rate-limiter is **not** a CV that mis-ranks configs — the leak-free CV is positively
order-consistent with the same-metric public LB and crowns the champion, which is genuinely public-
best pre-patch. The "CV up / public down" is (1) the external division-exploit **metric-patch** on the
frozen champion (0.624 ≡ 0.509, one artifact) plus (2) an internal **node-count-penalty** lever that
inflated the CV without improving matching. (1) is disclosed and un-chaseable; (2) is removed by the
re-anchored `lineage_macro_raw` KPI. Promotions now gate on penalty-free, lineage-parity matching
quality; the strict "order the three public points" target is inconclusive by construction and
recorded. → unblocks SOT-2895 (A/B on this re-anchored CV) and the parent's probe decision.
