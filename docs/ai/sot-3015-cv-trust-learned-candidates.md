# SOT-3015 — CV-trust report: hardening the leak-free CV for judging learned candidates

**Type:** enabling / role-B transfer-trust (no champion change, **no Kaggle submission**, no promotion,
**scorer untouched**). **Parent:** SOT-3010 (biohub-claude cycle-3, direction 4). **Builds on:** SOT-2761
(leak-free CV), SOT-2817/2903/2894 (re-anchor + transfer-trust), SOT-2929 (holdout grain), SOT-2995
(official-scorer fidelity), SOT-2993 (self-trained UNet, leak-free), SOT-3011 (released royerlab weights).
Champion `detect-link-dog-v4-shorttrack-motion-gain1` (`micro_adj 0.6760`) — **byte-frozen, untouched**.

**Reproduce:**
- `PYTHONPATH=src .venv/bin/python -m biohub_tracking.eval.leak_audit --out experiments/sot3015/leak_audit_report.json`
  → the machine-readable audit (pure, data-free).
- `PYTHONPATH=src .venv/bin/python -m pytest tests/test_leak_audit.py -q` → the audit logic (12 tests).
- `PYTHONPATH=src .venv/bin/python -m biohub_tracking.eval.transfer` and `... eval.holdout` → the
  current-CV vs reinforced-CV ranking-stability tables (pure; §3).

---

## 0. TL;DR

1. **The gap this closes.** The leak-free CV (`eval/cv.py`) is a faithful oracle **for the scorer**
   (SOT-2995: byte-exact to the official `tracksdata` scorer; SOT-2903: ρ=0.80 vs public over 4 anchors).
   But its leak-free guarantee is *conditional on nothing being fit on the holdout* — its own docstring
   says "there is nothing learned across time to leak." That is true for the **classical** champion and
   **false for a learned candidate** (directions 1-2). Scoring-side entity/temporal holdout protects the
   scorer; it does **not** protect a *learner*. A learned candidate needs a **training-side** holdout too.
2. **Ported external validation design.** The official baseline trains **per CV split** — the released
   weights are literally `weights/unet_transformer/split_0/…`: fit on a train split, validate on
   held-out videos it never trained on. The public EDA notebooks' recurring known-leak warning is
   **same-embryo / same-lineage** contamination (two videos of one developing embryo share appearance,
   illumination, developmental lineage).
3. **The concrete finding on our 4-family holdout.** SOT-2993's leak-free learned-detector A/B trained
   **leave-one-*video*-out** (4 folds, each on the other three videos). Every such fold still trains on
   the **same-embryo sibling** of the scored video (holding out `44b6_0113de3b` still trains on
   `44b6_0b24845f`) — exactly the same-lineage leak the notebooks warn about. The truly leak-free
   discipline is **leave-one-*lineage*-out** (2 folds: hold out **all** `44b6`, or **all** `6bba`).
4. **Provenance verdicts** (the CV-trust rule for the parent's two-signal gate): released-weights CV
   (SOT-3011 ILP 0.8081) is an **optimistic, train-contaminated upper bound** (split membership vs our
   holdout is not confirmable); leave-one-video-out CV (SOT-2993 0.6217) is a **weaker upper bound** (soft
   same-lineage leak); only **leave-one-lineage-out** CV is promotable leak-free evidence. The 0.62→0.81
   spread between SOT-2993 (leak-free retrain) and SOT-3011 (released weights) *is* that contamination gap.
5. **Ranking stability.** Current CV (`micro_adj`) and the reinforced/finer holdout statistics
   (`lineage_macro_adj` / `per_sequence_parity` / `regime_parity_adj`) crown the **same** config on the
   classical anchors and share the same Spearman (§3) — the reinforcement does not destabilise the
   ordering; it adds a *learner-side* guardrail the current CV structurally cannot express.

**Nothing here changes the scorer, the champion, or the live gate** (diff vs `main` = exports in
`eval/__init__.py` + the new `eval/leak_audit.py`, `tests/test_leak_audit.py`, this doc, the artifact).

---

## 1. External validation design vs our split (criterion 1)

Sources (read across SOT-2903/2995/3011; this issue extracts the **split** design, not the scorer —
scorer fidelity is already settled byte-exact in SOT-2995):

| Source | What it prescribes | Leak concern it encodes |
| --- | --- | --- |
| `github.com/royerlab/kaggle-cell-tracking-competition` (`scripts/train_unet_transformer.py`, released `…/split_0/edge_predictor_best.pth`) | Train the learned pipeline **per CV split**; validate on videos held out of that split. | train-on-scored-video (hard) — a learner must never be fit on the video it is scored on. |
| `thibautgoldsborough/cellmot-baseline-artifacts` (`split_0` weights + config) | Ships one split's weights; split membership vs the public/hidden test is **not independently confirmable**. | released-weights provenance (soft) — released weights may have trained on our holdout families. |
| Public EDA/scoring notebooks (pilkwang EDA/baseline, harshitsama scoring, xiaoleilian classical) | Emphasise the dataset is **few embryos × multiple videos**; two videos of one embryo are near-duplicates in appearance/illumination. | same-embryo / same-lineage (soft) — holding out one video while training on its sibling still leaks. |

**Diff against our split (`eval/cv.CV_HOLDOUT`).** Our holdout is a strict per-video partition (whole
`.geff` embryo videos) with **causal temporal** prediction — this closes the two *hard* entity leaks for
**any** candidate, and SOT-2995 closes scorer fidelity. What our split **did not** encode, because the
classical champion learns nothing, is the **training-side** discipline the external design assumes:
a learned candidate's model must be trained on a fold that excludes the scored video (hard), and — on a
few-embryos holdout — the scored video's **whole lineage** (soft). That is the enabling gap SOT-3015 fills.

## 2. Known-leak checklist, ported (criterion 1) — `eval/leak_audit.LEARNED_CANDIDATE_LEAK_CHECKLIST`

| id | severity | applies to | our status | cure |
| --- | --- | --- | --- | --- |
| `entity-holdout` | hard | both | **covered** | whole-video partition (`CV_HOLDOUT`); no cell/track/frame straddles a bucket. |
| `temporal-causality` | hard | both | **covered** | per-timepoint detection, forward-only linking; nothing fit across time. |
| `scorer-fidelity` | hard | scorer | **covered** | SOT-2995: `eval/official.py` == genuine `tracksdata` scorer, divergence 0 (8 golden + 8 real). |
| `train-on-scored-video` | hard | **learner** | **covered_for_scorer_only** | train each learned candidate on a fold that **excludes** the scored video (LOO). The classical champion needs no fit, so `cv.py` never had to enforce this — a learned candidate does. |
| `same-lineage-sibling` | soft | **learner** | **action_required** | train **leave-one-lineage-out** (hold out ALL `44b6` or ALL `6bba`), not leave-one-video-out; SOT-2993's per-video LOFO leaves the sibling in → soft leak. |
| `released-weights-provenance` | soft | **learner** | **action_required** | treat released-weights CV as an **optimistic upper bound**, never leak-free (SOT-3011 0.81 vs leak-free retrain SOT-2993 0.62). |

The first three are closed for the classical champion (and stay closed). The **crux** is
`covered_for_scorer_only` / `action_required`: those bite a *learned* candidate and were previously
implicit — this report makes them an explicit, unit-tested gate.

## 3. Ranking stability: current CV vs reinforced CV, same anchors, same seed (criterion 3)

The reinforced/finer-grain holdout statistics already exist (`eval/holdout.py`, `eval/transfer.py`); this
report re-runs them side by side over the historical classical lineage (v1 / dog-v2 / dog-v4-shorttrack,
the three public-anchored configs — v3-adaptive is `public_lb=None`, excluded to avoid mis-anchoring):

| statistic (grain) | Spearman ρ vs public (3 anchors) | crowns the public winner? |
| --- | ---: | :---: |
| `micro_adj` — **current CV / live KPI** (whole holdout) | 0.50 | ✅ |
| `lineage_macro_adj` — lineage parity (reinforced) | 0.50 | ✅ |
| `per_sequence_parity` — 4-video parity (reinforced, finest) | 0.50 | ✅ |
| `regime_parity_adj` — density-regime parity (reinforced, crosscuts lineage) | 0.50 | ✅ |
| `dense_regime_adj` — dense stratum only | 0.50 | ✅ |
| `sparse_regime_adj` — sparse stratum only | 0.50 | ❌ (fails to crown) |

**Reading.** The ρ=0.50 (not higher) is the well-known v1↔dog-v2 public near-tie (0.509 vs 0.500), not a
reinforcement artifact — it is identical across every grain. Every *parity/blended* reinforced statistic
crowns the same champion as the current CV → **the ordering is stable** under reinforcement. The only
statistic that *changes* the crown is the **single-stratum** sparse-regime view, which is strictly worse
(consistent with SOT-2929's conclusion: finer grain is leak-free but a *weaker* private proxy). So the
reinforced holdout is adopted as a **per-regime / per-lineage no-regression guardrail**, not a replacement
KPI — and the binding limitation stays **public-anchor scarcity**, not holdout grain.

**Why this matters for learned candidates.** All the anchors above are *classical*, so this table proves
the *scoring-side* ranking is stable; it cannot exercise the *learner-side* leak, because none of these
configs were trained on the holdout. That is exactly why the learner-side discipline (§4) is a separate,
new axis rather than another re-anchor of the scoring statistic.

## 4. Learned-candidate fold discipline + provenance verdicts (criterion 2 — "plug the leak")

`eval/leak_audit.learned_candidate_folds(by=…)` formalises the two disciplines (pure, unit-tested):

- **`by="lineage"` (recommended, leak-free):** 2 folds — `{hold 44b6_0113de3b + 44b6_0b24845f}` and
  `{hold 6bba_05b6850b + 6bba_05db0fb1}`. `residual_lineage_leak = False` for both.
- **`by="video"` (SOT-2993's scheme):** 4 folds — each holds one video and **trains on its sibling**.
  `residual_lineage_leak = True` for **all four**.

`train_val_leak_audit(train, val)` audits any declared split and reports **hard** (video in both sets)
vs **soft** (val lineage present in train) leak; `provenance_verdict(...)` returns the CV-trust class:

| candidate | how fit | verdict | CV-trust for promotion |
| --- | --- | --- | :---: |
| SOT-3011 released royerlab weights (ILP 0.8081) | external weights, split membership unconfirmed | `optimistic_upper_bound` | ❌ |
| SOT-2993 self-trained UNet, leave-one-**video**-out (0.6217) | local, holdout video excluded, **sibling kept** | `optimistic_upper_bound` (soft same-lineage leak) | ❌ |
| leave-one-**lineage**-out retrain (target) | local, whole scored lineage excluded | `leak_free_retrained` | ✅ |

So the promotable learned number does **not** yet exist: SOT-3011's 0.81 is contaminated, SOT-2993's
0.62 is a soft-leaky lower-ish bound. The 0.62→0.81 spread is the contamination+convergence gap
(consistent with SOT-3011's own leak caveat). **The next learned candidate must retrain
leave-one-lineage-out to produce a CV number the two-signal gate can trust.**

## 5. CV-trust decision rule for the parent-resume two-signal gate

When a direction-1/2 learned candidate reports a CV jump, gate it as:

1. **Provenance first** (`provenance_verdict`): released/external weights → *upper bound, not promotable
   on CV alone*; trained-on-scored-video → *reject*; leave-one-video-out → *upper bound*; only
   **leave-one-lineage-out** CV is promotable leak-free evidence.
2. **Then the existing scoring-side gate unchanged:** re-anchored `micro_adj` rises past noise **AND** 4/4
   per-dataset non-regression (`cv.CvResult.no_regression_vs`) **AND** the reinforced per-lineage/per-regime
   guardrail does not regress a stratum (§3), **AND** public LB does not contradict (rogii: public is a
   sparse falsifier only).
3. **A learned CV that clears (1) but only at leave-one-video-out grain is a HOLD**, not a promotion:
   its number is an upper bound, so a public LB probe carries proportionally more weight.

This report is the enabling artifact for that gate. It changes no scorer, no champion, no live threshold —
it supplies the **provenance + training-side leak** dimension the CV could not previously express.

Raw numbers: [`experiments/sot3015/leak_audit_report.json`](../../experiments/sot3015/leak_audit_report.json).
