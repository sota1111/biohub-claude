# SOT-3020 — learned-detection pipe: live-probe the real public LB

## Why (observation probe — NOT a champion promotion)

The 型E paradigm-shift built a **learned-detection substrate** candidate whose
*internal* leak-free CV looked strong:

| candidate | internal micro_adj | note |
| --- | --- | --- |
| SOT-3011 wholesale (released weights, learned detect + ILP link) | 0.8081 | REJECT 4/4 (44b6 sparse regressed) |
| SOT-3012 hybrid (learned detect substrate + champion classical link) | **0.8344** @thr0.99 | REJECT 4/4 (44b6 regressed) |
| champion (classical DoG + motion-link, gain=1.0) | 0.6760 | reigning, config `f2b1076…6522fc` |

But SOT-3015 showed the internal CV is **contaminated**: the released split_0
weights were trained on the labelled train videos, which *are* the four CV
families. True leak-free (leave-one-LINEAGE-out) is only **0.6217**; the
0.62→0.81 gap is contamination, not skill. So the internal jump may be an
artefact that does **not** transfer to the real (hidden-GT) test.

**The only uncontaminated signal is the live Kaggle public LB.** This probe
submits one learned-pipe run verbatim and reads the real public score to decide
between "internally high only because contaminated" vs "learned detection really
transfers → path A has large headroom".

## What was submitted

- **biohub is a Code (notebook-only) competition**: a raw `submission.csv` upload
  is rejected with HTTP 400 *"This competition only accepts Submissions from
  Notebooks."* So the probe had to be a **Kaggle notebook kernel**, not the
  locally-built CSV.
- Chosen candidate: the **SOT-3011 wholesale** learned pipeline (official royerlab
  offline inference notebook: learned TemporalUNet3D detection + ILP linking,
  `DET_THRESHOLD=0.99`, released split_0 weights). This is the explicitly-allowed
  fallback in the issue: the SOT-3012 hybrid needs a bespoke, Kaggle-untested
  kernel (embed royerlab detection + champion classical link), whereas the
  wholesale path is the organiser's own tested, offline, GPU submission notebook —
  far more reliable for a single measurement. Both answer the same question
  (does learned detection transfer?).
- Kernel: `sota1111/biohub-claude-learned-probe-sot-3020`
  (`submit/kernel-learned-probe/`), GPU on, internet off, weights from the
  attached `thibautgoldsborough/cellmot-baseline-artifacts` dataset. Paths are
  discovered robustly (pushed-kernel mounts differ from the interactive editor).
- Submitted **through the gate only**: `scripts/ai/kaggle_targets_submit.sh`
  (reserve/spacing/cap/fingerprint decided deterministically). No direct Kaggle
  API bypass. One submission only.

## Reading the result (input to the next cycle)

- Real public ≈ **0.62** → internal jump is contamination; learned detection does
  **not** transfer to the hidden test as-is (watch for learned-side overfit too).
- Real public **0.8-class** → learned detection **transfers**; path A (self-train
  a leak-free detector, leave-one-LINEAGE-out per SOT-3015, + fix 44b6
  over-detection) has real headroom for a 4/4-gate pass next cycle.

## Champion invariant

Champion stays the classical motion-link (CV 0.6760, `champion/config.json`
sha256 `f2b107674d870cfd8e1b667a5d487b15b994382f9de0e9c3bc66a0c05b6522fc`). This
probe is **not** promoted; the registry submit pointer is restored to the
champion kernel after the probe submission.

## Live result

<!-- filled in after the submission scores -->
- Public LB: _pending_
