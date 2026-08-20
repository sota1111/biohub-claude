# SOT-2840 — global min-cost-flow θ=6.5/window=2 LB-probe candidate artifact

**Parent:** SOT-2839 (biohub-claude Kaggle順位向上サイクル第4次)
**Related:** SOT-2830 (candidate origin), SOT-2827 (deferral), SOT-2817 (re-anchored CV)
**Outcome:** candidate artifact built with a **new fingerprint**; live champion **byte-frozen**;
**no Kaggle submission** (handed to the parent resume run's reserve-slot LB probe).

## Goal

SOT-2830 found the project's **first** non-regressing, all-aggregation-consistent linking
candidate — the portable global short-window min-cost-flow / birth-death-arc linker at
**θ=6.5 / window=2** (CV 0.6649→0.6671, 4/4 per-dataset non-regression) — but it is
`family_mix_sensitive=True`, so SOT-2827/2839 deferred any champion pointer flip until a real
**LB probe** could resolve whether the dominant-6bba CV gain transfers. This issue builds that
candidate as a **submittable artifact with a distinct fingerprint** so the parent resume run can
probe the LB via the **reserve slot** — **without** flipping the live champion prematurely
(SOT-2817 representativeness guard / SOT-2816 CV-up→LB-down hazard).

## What was built

1. **Candidate config** `champion/candidates/sot2840-globalmcf-theta6p5-window2.json` — identical
   to the reigning `detect-link-dog-v4-shorttrack` champion **except** a linking-only delta:
   `global_window=2`, `birth_cost=death_cost=3.25` (θ=birth+death=6.5). Detection knobs unchanged.
2. **Candidate build path** in `submit/build_kernel.py` — `--candidate-config <json>` builds an
   LB-probe kernel to `submit/kernel-candidate/` under a **distinct kernel id**
   `sota1111/biohub-claude-candidate`. The generated kernel embeds the candidate config and, at
   runtime, materialises it and points `BIOHUB_CHAMPION_CONFIG` at it **for its own process only**
   — the live champion config/kernel is never mutated. The champion build path is byte-unchanged
   and deterministic (regression-tested).

## Verification (`experiments/sot2840/confirm_candidate_artifact.py`, PASSED)

| check | result |
| --- | --- |
| effective config | `global_window=2, birth=death=3.25`, detection == champion ✅ |
| candidate CV score | **0.6671** (== SOT-2830) ✅ |
| aggregations | micro 0.6649→**0.6671**, macro 0.7180→**0.7233**, lineage-macro 0.7216→**0.7272** — all up ✅ |
| per-dataset non-regression (4/4) | 0.8895→0.8906, 0.6817→0.6970, 0.5700→0.5717, 0.7310→0.7339 ✅ |
| family_mix_sensitive | **True** (re-confirmed; dominant-6bba gain) ✅ |
| champion default reproduced | **0.6649** byte-for-byte (`eval.cv --check-champion`, Δ 0.0000) ✅ |

### Fingerprints

| artifact | sha256 | note |
| --- | --- | --- |
| live `champion/config.json` | `42064648…e01bdd` | **BYTE-FROZEN** (== last-submitted, public 0.509) |
| candidate config | `55af7cc8…f8e6c0` | NEW (≠ champion) |
| candidate kernel `submit/kernel-candidate/biohub-claude-candidate.py` | `3f955095…4cc1c0` | NEW (≠ champion kernel) |
| committed champion kernel | `48b1eaa2…1ee3a698` | untouched |

### Gates
- `pytest` **156 passed** (incl. `tests/test_build_kernel_candidate.py`, 4 new)
- `python -m compileall -q src scripts submit` OK
- `python submit/exec_compat_gate.py` OK

## Handoff to the parent resume run (SOT-2839)

- Push `submit/kernel-candidate` (kernel id `sota1111/biohub-claude-candidate`) and submit it via
  the **reserve slot** to LB-validate the θ=6.5/window=2 candidate.
- The candidate fingerprint differs from the last-submitted champion, so the improvement-gate /
  fingerprint gate will not skip it.
- **Do NOT flip the champion pointer** until the LB confirms the family-mix-sensitive CV gain
  transfers. On confirmation, promote the candidate config to `champion/config.json` (+ embedded
  copy) in a follow-up; on non-transfer, record the LB result and keep the champion.
