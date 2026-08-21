# SOT-2900 — Motion-model-link parameter tuning on leak-free CV

Parent: SOT-2897 (biohub-claude Kaggle 順位向上サイクル 第3次). **Primary KPI = leak-free CV**
(SOT-2761 entity holdout, SOT-2817 re-anchored full metric). **No Kaggle submission in this Issue.**

## Goal

Systematically tune the internal knobs of the one robust promotion lever in this competition —
SOT-2864's ARGUS-style motion-model predicted-position LAP linking (`motion_model_link`) — to beat its
default leak-free CV micro-adjusted edge Jaccard of **0.6760** while keeping **4/4 per-dataset
non-regression** (delta-family-robust, not family-mix-sensitive).

Baseline candidate (SOT-2864 / SOT-2896): `motion_model_link=true, motion_smooth_sigma=15.0,
motion_gain=1.0, motion_gate_on_prediction=true, velocity_disp_weight=0.05`; detect knobs identical to
the byte-frozen champion (`champion/config.json`, sha256 `42064648…`, CV micro 0.6649).

## Method

Single-variable A/B on the leak-free CV harness (`biohub_tracking.eval.evaluate_cv(config=…)`), same
seed / deterministic pipeline, one full 4-video CV per config (~120 s). 32 configs total.

## Results

`motion_gain` is by far the strongest lever — a smooth **unimodal** curve, peaking on a broad flat
plateau then collapsing:

| motion_gain | 0.5 | 0.85 | 1.0* | 1.5 | 1.75 | 2.0 | 2.25 | 2.35 | 2.5 | 3.0 | 4.0 | 6.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CV micro | 0.6722 | 0.6747 | **0.6760** | 0.6786 | 0.6805 | 0.6821 | 0.6830 | 0.6840 | 0.6792 | 0.6555 | 0.5859 | 0.4106 |
| 4/4 non-reg | REG | OK | OK | OK | OK | OK | OK | OK | OK | REG | REG | REG |

\* = SOT-2864 default baseline.

- **`motion_gate_on_prediction=true` is essential** — gate-off at gain=2.0 collapses to 0.6687 (REG).
- **Secondary knobs add ~nothing beyond gain**: best `motion_smooth_sigma`=18 (+0.0009 over base),
  `velocity_disp_weight` flat within ±0.0007 (best 0.1, +0.0007); every combo ≤ the pure-gain gain.
- Reducing gain below 1.0 or raising sigma too high regresses the clean 44b6_0113de3b family.

## Selected candidate — `motion_gain = 2.0`

Chosen on the **plateau center** as the roundest, least-overfit value. The empirical argmax gain=2.35
scores only ~0.002 higher (≈1 edge on a 4-video holdout — within noise), so 2.0 is the deliberately
conservative pick over argmax-overfitting the tiny holdout.

| metric | champion | SOT-2864 (gain 1.0) | **SOT-2900 (gain 2.0)** |
|---|---|---|---|
| CV micro-adj | 0.6649 | 0.6760 | **0.6821** (+0.0172 / +0.0061) |
| macro-adj | 0.7180 | 0.7310 | **0.7575** |
| lineage-macro | 0.7216 | 0.7351 | **0.7623** |
| 44b6_0113de3b | 0.8895 | 0.9078 | 0.9835 |
| 44b6_0b24845f | 0.6817 | 0.6938 | 0.7155 |
| 6bba_05b6850b | 0.5700 | 0.5748 | 0.5739 |
| 6bba_05db0fb1 | 0.7310 | 0.7477 | 0.7569 |

**4/4 per-dataset non-regression**, and micro + macro + lineage-macro all rise together → **not
family-mix-sensitive** (the gain is not concentrated in the dominant lineage).

## Artifact & fingerprint

- Candidate config: `champion/candidates/sot2900-motion-model-link-gain2.json`
  — effective-config fingerprint **sha256 `5318dbb2ab654b42e2ec37ede6bb558902ede0b207058c0a1c7fddb2d1890660`**.
- Loads and reproduces CV micro 0.6821; builds an exec-compat-green candidate kernel via
  `python submit/build_kernel.py --candidate-config champion/candidates/sot2900-motion-model-link-gain2.json --out-subdir kernel-candidate-motion-gain2 --kernel-id sota1111/biohub-claude-candidate-motion-gain2`.
- **Live champion `champion/config.json` stays BYTE-FROZEN** (`eval.cv --check-champion` delta 0.0000).

## No submission — parent decides

Per the documented CV↔LB divergence (SOT-2816: a CV +0.042 mapped to public −0.115) and the SOT-2817
representativeness guard, a CV gain alone does not flip the live champion. This is a **candidate artifact
only**. The parent cycle (SOT-2897) owns the reserve LB-probe decision between this gain=2.0 candidate
and the existing SOT-2864 gain=1.0 candidate (`champion/candidates/sot2864-motion-model-link.json`).
