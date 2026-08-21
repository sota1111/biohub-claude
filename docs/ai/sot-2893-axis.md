# SOT-2893 — cycle-2 axis selection & decomposition (biohub-claude)

**Run:** initial cron dispatch (2026-08-21). **No Kaggle submission. Champion byte-frozen.**
Champion `champion/config.json` sha256 `42064648e612183e761bf9d40b70d3e8a2497453a878f1a44f5b52e410e01bdd`
(== last-submitted artifact, public **0.509**).

## Why this axis (stuck-detection + prior handoff)

- **Real rate-limiter = CV↔public NEGATIVE transfer**, not local saturation. Public regressed
  **0.624 (08-03) → 0.557 → 0.509 (08-20 champion)** while local leak-free CV reported "improvement"
  (SOT-2816: promoted CV +0.042 → public −0.115).
- Local A/B is exhausted: detection saturated cycle 2–9 (sparse-GT over-count penalty, single global
  operating point impossible); every linking lever terminal except **SOT-2864 motion-model linking**
  (leak-free CV 0.6649→0.6760, 4/4 families non-regressing, NOT family-mix-sensitive, **unsubmitted**).
- Escalation ladder forced to **step-2/3** (data/oracle re-anchor, generalization-gap diagnosis) and
  **step-6** (external-knowledge port), per the issue's stuck-detection banner and the SOT-2882 handoff.

## Web research (external knowledge — sources recorded to `experiment_ledger.jsonl`)

- Competition: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development
- Official baseline (TemporalUNet3D + SimpleNodeTransformer): https://github.com/royerlab/kaggle-cell-tracking-competition
  — learned-detector port already **REJECTED** (SOT-2848), nnPU **inconclusive** (SOT-2863).
- Forum: https://forum.image.sc/t/biohub-cell-tracking-during-development-kaggle-competition/121671
- Public Code-tab notebooks queued for SOT-2895:
  - xiaoleilian "Classical Baseline": https://www.kaggle.com/code/xiaoleilian/biohub-cell-tracking-classical-baseline
  - dalloliogm "Suspicious Tracking Event Review": https://www.kaggle.com/code/dalloliogm/biohub-suspicious-tracking-event-review
  - kaiwalyaatulraut: https://www.kaggle.com/code/kaiwalyaatulraut/biohub-cell-tracking-solution
  - anhadmahajan06: https://www.kaggle.com/code/anhadmahajan06/biohub-track-your-cells-development
- Shake-up lesson (public-LB overfitting via semi-automatic tuning → private drop):
  https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-tips-tricks-to-survive-a-kaggle-shake-up-23675beed05e

## Children (all Todo — no submission by children)

| Child | Axis | Ladder | Notes |
| --- | --- | --- | --- |
| SOT-2894 | Re-anchor leak-free CV to CV↔public negative transfer; order the 3 historical submissions (0.624/0.557/0.509) | step-2/3 | Gate: no promotion decision until CV re-anchored. Champion byte-frozen. |
| SOT-2895 | Investigate this competition's public notebooks/top solutions; port ≥1 non-rejected portable technique (candidate: dalloliogm-style post-hoc suspicious-track anomaly review) | step-6 | A/B on SOT-2894's re-anchored CV. **blockedBy SOT-2894.** default-off. |
| SOT-2896 | Build SOT-2864 motion-model linking (CV 0.6760) as a distinct-fingerprint submittable candidate for the parent's due probe slot | probe prep | Champion byte-frozen; child does NOT submit. |

Not a blind retry: rejected axes (nnPU / ARGUS / ultrack multi-hypothesis / gap-closing /
learned-detector / learned-gate / bidirectional-consistency) are not re-tried without new grounds.
SOT-2895's target is a distinct post-hoc, track-level anomaly family.

## Terminal for this run

Children registered in **Todo**; parent → **In Review** to await the webhook `auto-parent-resumed`
comment. The resume run aggregates all children and makes the submission decision (probe slot is due:
last submit 08-20 15:22Z, 4 effective slots) via
`scripts/ai/kaggle_targets_submit.sh --competition biohub --repo biohub-claude --execute`.
