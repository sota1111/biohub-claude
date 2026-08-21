# SOT-2864 — ARGUS motion-model predicted-position LAP linking (cycle-9 child B)

**Axis.** Port the core of ARGUS (arXiv:2607.08297, CTC 0.90–0.97, CPU-only / no
weights): match `t → t+1` against **where each cell is predicted to move** under a
motion model, instead of the champion's raw-distance nearest-neighbour. Implemented
as a **default-off** `motion_model_link` path in `src/biohub_tracking/link.py`.

## Mechanism (numpy/scipy-only, portable)

`cv2` (Farneback dense optical flow) is **not** installed in this repo / the offline
Kaggle kernel, so — per the acceptance criteria — the first-class path is a
numpy/scipy-only **detection-cloud motion field** (`_motion_field_predict`):

1. A provisional optimal assignment `src → dst` within `max_distance` yields *anchor*
   displacements `v_k = dst[j] − src[i]` at each matched source.
2. Each source's predicted displacement = anchor displacements weighted by a Gaussian
   of the scaled distance to each anchor source (`exp(−½ (d/σ)²)`, σ =
   `motion_smooth_sigma` µm). This diffuses the sparse anchor flow into a dense,
   locally-consistent field, so **every** source (history-less cells included) gets a
   prediction.
3. Final LAP is solved on the predicted positions `src + motion_gain · field`
   (reusing `_assign`'s `src_pred` path; gate via `motion_gate_on_prediction`).

Pure `numpy`/`scipy`, deterministic, no weights/internet — the exec-compat gate
(`submit/exec_compat_gate.py`) is green. A `cv2`-based Farneback estimator can drop
into step 1 where available; the fallback carries the mechanism portably.

**Distinct from prior linking axes.**
- Constant-velocity `velocity_gain` (SOT-2369) predicts each cell from *its own*
  `t−1 → t` edge → no prediction for a first-appearance cell. The motion field
  predicts *every* source from its neighbourhood **within the current frame pair**.
- Static gap-closing (SOT-2763, rejected) and node-interp gap-recovery (SOT-2849,
  rejected) act on missing/gap frames; this changes the **primary `t → t+1` cost**.

## A/B result (same-seed, leak-free SOT-2761 CV, champion classical detection)

Measured on the **champion classical detector** (learned detection SOT-2848 is
degenerate, micro-adj 0.0 — a linking A/B is only meaningful on the champion
detector, as SOT-2849 established). Detection cached once per family; every variant
re-linked off the cache and scored through the one CV aggregation ⇒ byte-comparable
to the registry champion CV (0.6649) = a same-seed A/B (deterministic pipeline).

Baseline champion reproduced **exactly**: micro-adj 0.6649 / score 0.6649.

Grid `motion_smooth_sigma ∈ {8,15,30} × motion_gain ∈ {0.5,1.0} × motion_gate_on_prediction ∈ {F,T}`
(`experiments/sot2862b/screen_motion_link.json`). **Best (and only) promotable
variant: σ=15 µm, gain=1.0, gate_on_prediction=True.**

| dataset | champion adj | motion adj | Δ |
| --- | --- | --- | --- |
| 44b6_0113de3b | 0.8895 | **0.9078** | +0.0183 |
| 44b6_0b24845f | 0.6817 | **0.6938** | +0.0121 |
| 6bba_05b6850b | 0.5700 | **0.5748** | +0.0048 |
| 6bba_05db0fb1 | 0.7310 | **0.7477** | +0.0167 |

- **micro-adj 0.6649 → 0.6760 (+0.0111)**; total edges ΔTP **+16** / ΔFP **−20** /
  ΔFN **−16**.
- macro-adj 0.718 → 0.731; lineage-macro 0.7216 → 0.7351 — micro **and** both
  family-mix-robust macro views rise together ⇒ **not family-mix-sensitive**.
- **Passes the per-dataset no-regression gate.** This is the **first linking axis in
  the ledger to improve all four families with zero per-dataset regression** —
  including the ultra-sparse 44b6_0113de3b that recovered **0** edges under
  gap-recovery (SOT-2849) and regressed every prior linking axis.

`motion_gate_on_prediction=True` is the discriminating knob: gating on the predicted
distance lets a fast, motion-consistent cell keep a link the raw-distance gate would
have dropped (the FN/FP cleanup above); with `gate=False` the field only re-ranks the
champion feasible set and the gain shrinks to ≤ +0.0021.

## Decision — CV-PROMOTABLE / MECHANISM-VALIDATED, **champion byte-frozen, shipped default-off, NO submission**

Despite the clean, per-dataset non-regressing CV gain, the champion pointer is **not
flipped** and **nothing is submitted** this cycle:

- **Documented CV↔LB divergence (SOT-2816):** the last champion promotion raised CV
  0.6232 → 0.6649 (+0.042) while the public LB *fell* 0.624 → 0.509 (−0.115). A CV
  gain has demonstrated **negative** correlation with the hidden LB here, so CV alone
  is not a safe basis to flip the byte-frozen champion (== last-submitted public
  0.509).
- **Mode / deadline:** competition deadline 2026-09-29 (~39 days out) ⇒ `mode=improve`,
  not converge/endgame; the daily reserve LB slot is conserved for the endgame.
- Issue acceptance mandates **no Kaggle submission** this run, and
  `champion/config.json` updates only on **昇格確認** (LB-confirmed promotion).

**Outcome:** `motion_model_link` shipped **default-off** (champion
`detect-link-dog-v4-shorttrack` byte-frozen, sha256
`42064648…e01bdd`, CV `--check-champion` delta 0.0000, full suite 178 pass).
Recorded as the **#1 reserve / final-window LB-probe candidate**, superseding the
prior queue (SOT-2849 gap-recovery = +0.0124 micro but *family-mix-sensitive* / 44b6
recovers 0; SOT-2840 global-MCF = +0.0022): this is the only linking gain that is
both positive **and** per-dataset non-regressing **and** not family-mix-sensitive.

**Candidate operating point (NOT promoted; default-off):** champion `link` block +
`motion_model_link:true, motion_smooth_sigma:15.0, motion_gain:1.0,
motion_gate_on_prediction:true`.

Artifacts: `experiments/sot2862b/screen_motion_link.{py,json,log}`;
`tests/test_motion_link.py` (5 tests). No Kaggle submission.
