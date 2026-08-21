# SOT-2896 — SOT-2864 motion-model-link submittable candidate artifact (cycle-2 child 3)

**Parent:** SOT-2893 (biohub-claude Kaggle順位向上サイクル第2次)
**Related:** SOT-2864 (candidate origin — motion-model LAP linking), SOT-2840 (candidate build path), SOT-2816/2817 (CV↔LB transfer / representativeness guard)
**Outcome:** candidate artifact built with a **new fingerprint**; live champion **byte-frozen**;
**no Kaggle submission** (handed to the parent resume run's due probe slot).

## Goal

SOT-2864's ARGUS-style motion-model predicted-position LAP linking is the project's **only**
linking lever that is simultaneously (a) a positive leak-free-CV gain (micro-adj 0.6649→0.6760,
**+0.0111**), (b) per-dataset non-regressing on all 4 LOFO families, and (c) delta-robust across
aggregation views (micro **and** macro **and** lineage-macro all rise together — *not*
family-mix-sensitive in the delta sense). It shipped **default-off** and was never submitted. This
issue materialises it as a **submittable artifact with a distinct fingerprint** so the parent resume
run can probe the LB via the **reserve slot** — **without** flipping the live champion prematurely
(SOT-2817 representativeness guard / SOT-2816 CV-up→LB-down hazard). **This child does not submit.**

## What was built

1. **Candidate config** `champion/candidates/sot2864-motion-model-link.json` — identical to the
   reigning `detect-link-dog-v4-shorttrack` champion **except** a linking-only delta:
   `motion_model_link=true, motion_smooth_sigma=15.0, motion_gain=1.0,
   motion_gate_on_prediction=true`. Detection knobs unchanged.
2. **Candidate build** via the existing SOT-2840 `--candidate-config` path in
   `submit/build_kernel.py`, written to its own kernel id / dir so it never overwrites the reigning
   champion **or** the SOT-2840 global-MCF candidate:
   `python submit/build_kernel.py --candidate-config champion/candidates/sot2864-motion-model-link.json --kernel-id sota1111/biohub-claude-candidate-motion --kernel-title "Biohub Claude Candidate (SOT-2864 motion-link)" --out-subdir kernel-candidate-motion --code-file biohub-claude-candidate-motion.py`
   → `submit/kernel-candidate-motion/biohub-claude-candidate-motion.py` (+ `kernel-metadata.json`,
   GPU/internet off). The generated kernel embeds the candidate config and, at runtime, materialises
   it and points `BIOHUB_CHAMPION_CONFIG` at it **for its own process only** — the live champion
   config/kernel is never mutated.

## Verification (`experiments/sot2896/confirm_candidate_artifact.py`, PASSED)

| check | result |
| --- | --- |
| effective config | `motion_model_link=True, sigma=15.0, gain=1.0, gate=True`, detection == champion ✅ |
| candidate CV score | **0.6760** (== SOT-2864, tol 1e-3) ✅ |
| aggregations | micro 0.6649→**0.6760**, macro 0.7180→**0.7310**, lineage-macro 0.7216→**0.7351** — all up ✅ |
| per-dataset non-regression (4/4) | 0.8895→0.9078, 0.6817→0.6938, 0.5700→0.5748, 0.7310→0.7477 ✅ |
| delta family-robust | **True** (all 3 aggregation views rise together — the SOT-2864 non-sensitivity) ✅ |
| champion default reproduced | **0.6649** byte-for-byte (`eval.cv --check-champion`, Δ 0.0000) ✅ |

### The `family_mix_sensitive` flag (important nuance)

`representativeness_report.family_mix_sensitive` is a pure **absolute-gap** flag:
`|micro − lineage_macro| > mix_tol(0.05)`. On this holdout the dominant 6bba lineage carries 95.8%
of the micro weight, so the absolute micro↔lineage-macro gap is ~0.057–0.059 **for the champion
itself** (0.6649 vs 0.7216 → 0.0567 > 0.05 → flag True). The flag is therefore an inherent property
of the family weighting, **not** of a candidate's delta, and is `True` for this candidate too
(0.6760 vs 0.7351 → 0.0591). SOT-2864's documented **"not family-mix-sensitive"** is the *delta*
notion — the improvement is not concentrated on one family (micro + macro + lineage-macro all rise
together, no per-dataset regression). That delta-robustness (`delta_family_robust` above) is what
distinguishes SOT-2864 from the SOT-2840 global-MCF candidate, whose gain was 6bba-concentrated. The
confirm script gates on the delta notion and records the absolute flag as informational.

### Fingerprints

| artifact | sha256 | note |
| --- | --- | --- |
| live `champion/config.json` | `42064648…e01bdd` | **BYTE-FROZEN** (== last-submitted, public 0.509) |
| candidate config | `ca4da143…54cf340` | NEW (≠ champion) |
| candidate kernel `submit/kernel-candidate-motion/biohub-claude-candidate-motion.py` | `e9de1f9c…99ddeb49` | NEW (≠ champion kernel, ≠ SOT-2840 candidate) |
| committed champion kernel | `48b1eaa2…1ee3a698` | untouched |

### Gates
- `pytest` **225 passed** (incl. `tests/test_build_kernel_candidate.py`)
- `python -m compileall -q src scripts submit` OK
- `python submit/exec_compat_gate.py` OK (`nodes=10 edges=8`)
- Kaggle **NOT submitted** (parent resume run's gated `kaggle_targets_submit.sh` decides).

## Handoff to the parent resume run (SOT-2893)

- Push `submit/kernel-candidate-motion` (kernel id `sota1111/biohub-claude-candidate-motion`) and
  submit it via the **reserve slot** to LB-validate the SOT-2864 motion-link candidate. Its
  fingerprint differs from the last-submitted champion, so the improvement/fingerprint gate will not
  skip it.
- Priority over the SOT-2840 global-MCF candidate: SOT-2864 is +0.0111 (vs +0.0022) and delta
  family-robust (vs 6bba-concentrated / family-mix-sensitive) — the **#1 reserve / final-window
  LB-probe candidate**.
- **Do NOT flip the champion pointer** until the LB confirms the CV gain transfers (documented
  negative CV↔public transfer SOT-2816). On confirmation, promote the candidate config to
  `champion/config.json` (+ embedded copy) in a follow-up; on non-transfer, record the LB result and
  keep the champion.
