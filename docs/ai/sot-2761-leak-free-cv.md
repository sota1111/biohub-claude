# SOT-2761 — Leak-free CV & the pre-submission checklist

Kaggle rank-improvement cycle 2 (parent **SOT-2757**), validation-discipline axis.
**No Kaggle submission was made; no champion state was mutated.**

This issue promotes the four-dataset holdout that SOT-2305→2369 built up into the
repo's **one primary-KPI CV**: a single reusable evaluation function every later
child Issue screens and confirms against, plus the written record of *why it is
leak-free* and *how it maps to the public LB*, so no future cycle silently drifts
its oracle again.

## The CV: `biohub_tracking.eval.cv`

`src/biohub_tracking/eval/cv.py` is the single source of truth for the CV. It
exposes:

| symbol | role |
| --- | --- |
| `CV_HOLDOUT` | the four holdout entities (name, lineage, test image, train GT) |
| `evaluate_cv(config=None)` | run the champion (or any config) over the holdout → `CvResult` |
| `score_family(...)` / `aggregate(...)` | pure per-family scoring + micro-averaging (data-free) |
| `CvResult.no_regression_vs(incumbent)` | the mandatory per-family selection gate |
| `cv_result_to_dict(...)` | JSON-serialisable report view |

`evaluate_cv()` returns the **micro-adjusted edge Jaccard + per-dataset breakdown
+ division Jaccard** (and a per-lineage breakdown). It replaces the copy-pasted
`HOLDOUT = [...]` list and hand-rolled micro-averaging that previously lived in
`scripts/reanchor_oracle.py`, `experiments/sot2369/confirm_shorttrack.py`, and
each `screen_*.py` — the exact duplication through which an oracle drifts between
screens. Aggregation is byte-identical arithmetic to `eval/score.py`
(sum TP/FP/FN for the micro Jaccards; weight-average the per-family adjusted
Jaccard by `w = tp+fp+fn`), so the champion still reproduces its registry number.

CLI:

```bash
.venv/bin/python -m biohub_tracking.eval.cv --check-champion --out experiments/sot2761/cv_champion.json
```

`--check-champion` fails (exit 1) if the champion no longer reproduces the
registry reference micro-adj **0.6649** within tolerance — a standing guard
against silent CV/pipeline drift.

## Why it is leak-free

The competition scores a hidden **private** split; the checklist below exists so
we never trust a leaky CV. Two boundaries matter, and neither leaks here:

### 1. Entity holdout (embryo-lineage, not row/frame)

The scored unit is a **whole embryo video** (a `.geff` lineage), never an
individual frame or node, so no cell and no track straddles the train↔score
boundary. The holdout is exactly the four families the Kaggle test set scores:

| family | lineage (entity) | scored input | ground truth |
| --- | --- | --- | --- |
| `44b6_0113de3b` | `44b6` | `data/test/44b6_0113de3b.zarr` | `data/train/44b6_0113de3b.geff` |
| `44b6_0b24845f` | `44b6` | `data/test/44b6_0b24845f.zarr` | `data/train/44b6_0b24845f.geff` |
| `6bba_05b6850b` | `6bba` | `data/test/6bba_05b6850b.zarr` | `data/train/6bba_05b6850b.geff` |
| `6bba_05db0fb1` | `6bba` | `data/test/6bba_05db0fb1.zarr` | `data/train/6bba_05db0fb1.geff` |

Two distinct embryo lineages (`44b6`, `6bba`), two videos each. The GT `.geff`
comes from the **train** split (which ships GT for all four test videos) and is
used **only to score**, never to fit — the detector/linker is a deterministic
classical pipeline with hand-set parameters, so there is no model into which a
label could leak. The scored *image* is the actual **test** volume, so the CV
target **is** the LB target (SOT-2305 re-anchoring).

### 2. Temporal holdout (forward-only, causal)

Within each video, frames are time-ordered `t = 0..T`. Detection is
**per-timepoint** and linking is **forward-only** (`t → t+1`,
`biohub_tracking.link.link_centroids`), so no future frame informs an earlier
prediction and no parameter is fit on future data. There is nothing learned
across time to leak; "hold out the future" is satisfied structurally by the
pipeline being causal and deterministic (no RNG — re-running reproduces every
score bit-for-bit).

### 3. Selection discipline (the residual risk)

With only **four** videos, the real leakage risk is not row leakage but
**selection overfitting**: repeatedly tuning a knob on the same four-video micro
until it looks good. Mitigation, enforced by the harness and required of every
child Issue: gate a promotion on **per-dataset no-regression**
(`CvResult.no_regression_vs`) *and* the per-lineage breakdown, not the micro
alone — a micro gain that regresses any single family/lineage is rejected. This
is exactly the rule the champion promotions already followed (mtl=5 was rejected
for regressing the clean `44b6_0113de3b` family despite a higher micro).

## CV ↔ public-LB order consistency

Checklist item 2 asks whether the CV is the **same order** as the LB (a
digit/order-of-magnitude gap means leak or CV-design error).

| quantity | value |
| --- | --- |
| champion CV micro-adj (this harness) | **0.6649** |
| champion public LB best | **0.624** |
| gap | +0.041 (CV mildly optimistic, **same order**) |

Same order, small positive gap. The mild optimism is expected and benign: the CV
scores against the train-split GT of the four test videos, i.e. a full-GT view of
the same volumes the LB scores on a hidden subset — so a small CV≥LB gap is the
normal train-GT-vs-hidden-subset difference, **not** a leak. Per the checklist,
had they diverged in order (or had CV≫LB) we would suspect a leak / CV-design
mistake and fix the CV before trusting any public number. They do not, so the CV
is accepted as the primary KPI; public LB stays the secondary check.

Residual caveat inherited from SOT-2305 (unchanged, cannot be closed without a
submission this cycle forbids): the holdout ranks the DoG family well above the
old `v1` global-threshold detector, whereas on the LB they were near-tied — so
the CV is a faithful **champion-tracking** oracle but not a perfect *cross-family*
ranker. It is the best available oracle and every promotion also checks
per-dataset no-regression to stay honest.

## Pre-submission checklist — status in this repo

Walking `docs/kaggle-playbook/README.md` → 提出前チェックリスト against biohub-claude:

- [x] **leak-free 検証があるか（エンティティ hold out / 時系列は未来を hold out）** — yes: `CV_HOLDOUT` is an embryo-lineage entity holdout; the pipeline is causal/forward-only (§"Why it is leak-free").
- [x] **CV は LB と同じオーダーか（桁ズレならリーク/CV設計ミスを疑う）** — yes: CV 0.6649 vs public 0.624, same order (§"CV ↔ public-LB").
- [x] **public と CV が乖離したら悲観的な CV を信じる** — policy recorded: CV is the primary KPI, public is secondary; on divergence trust the (pessimistic) CV and fix it before chasing public. Enforced by promoting only on the CV + per-dataset no-regression gate.
- [x] **metric は重い裾か → 頑健受容テスト** — the score is a bounded Jaccard in [0, 1] (edge Jaccard + 0.1·division Jaccard, node-count penalty floored at 0), **not** a heavy-tailed RMSE/log metric, so a single catastrophic case cannot dominate. The per-dataset no-regression gate is the robustness test (no family may regress).
- [x] **公開NB/参照実装の public を上回ったら後付け較正の過学習を疑う** — n/a in the leaky direction: the champion (0.624) is **below** the public frontier notebook (0.913, GPU UNet+ILP), and that notebook's gain cannot run in this CPU/no-weights kernel; only its classical `FILTER_SHORT_TRACKS` lever was ported and it was CV-confirmed (SOT-2369). We are not over-claiming a public over-performance.
- [x] **最終提出枠は CV最良 × hedge で分散** — noted for the converge/submission child: select final slots by CV-best plus a hedge; do not put all slots on the single public-best kernel. (No submission this cycle.)
- [x] **Code コンペなら exec 互換ゲート + ランタイム前提** — `submit/exec_compat_gate.py` + `tests/test_exec_compat.py` enforce the champion runs with no `__file__` / no filesystem and that `EMBEDDED_CHAMPION_CONFIG == champion/config.json`; pipeline deps are numpy/scipy/zarr only.
- [x] **提出には `[repo:...] [lineage:...]` マーカー** — attribution markers are applied by the control-plane `kaggle_targets_submit.sh` at submission time (see cycle-4 submit ref 55193790). Nothing to fill in-repo.

No previously-unmet checklist item required a code change beyond wiring the CV
harness this issue adds; the checklist is now traceable to concrete repo
artifacts above.

## Reproduce

```bash
pip install -e .
python -m pytest -q                              # 63 tests incl. tests/test_cv_harness.py
python -m biohub_tracking.eval.cv --check-champion --out experiments/sot2761/cv_champion.json
```

The champion CV reproduction lands in `experiments/sot2761/cv_champion.json`
(micro-adj 0.6649, per-dataset + per-lineage breakdown, `cv_public_order_consistent: true`).
Deterministic (no RNG) — re-running reproduces every number.
