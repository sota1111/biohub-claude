# SOT-2882 — cycle-8 axis selection (biohub-claude)

**Banner:** stuck-detection (flat LB rank) → external-knowledge escalation forced
(ladder step-6). Research this competition's top solutions **and** its own public
notebooks; port ≥1 portable method. **Initial decomposition run — no Kaggle
submission; children remain `Todo`; parent waits `In Review`.**

## Situation (from the ledger)

- Champion `detect-link-dog-v4-shorttrack`: DoG-v3 adaptive detection + **distance-only
  NN** linking + short-track prune (`min_track_length=4`), `allow_division:false`.
  Leak-free micro-adj CV **0.6649**, last-submitted **public 0.509** (byte-frozen).
- **Detection is saturated**: sparse-GT PU-contamination *confirmed* (SOT-2863); every
  operating-point + learned-detector axis rejected across cycles 2–9
  (multiscale-DoG / quantile-norm / watershed / density-split / local-MAD /
  Hessian-blobness / learned-scorer / U-Net / nnPU / recall-recovery).
- **Linking is the one productive stage.** SOT-2864 motion-model linking is the only
  robust lever: **0.6649 → 0.6760 (+0.0111)**, first & only linking axis to improve
  **all four** LOFO families with **zero per-dataset regression**, *not*
  family-mix-sensitive. It was **never submitted** (see gap below).
- **Binding constraint — negative CV↔public-LB transfer.** SOT-2816 had CV **+0.042**
  while public LB moved **−0.115**; SOT-2864's robust +0.0111 CV gain is therefore held
  unsubmitted. Doctrine: **trust leak-free CV, do not chase public.** Guard: every child
  A/B requires **all-4-family non-regression** so a promoted lever is maximally likely
  to transfer.

## External research (2026-08-21)

- **No winner/medal write-ups exist** (competition still active, launched 2026-06-29).
  The competition's own public notebooks are baseline-derivative; Kaggle NB bodies are
  JS-rendered (unreadable via fetch) — consistent with SOT-2866.
- **royerlab official baseline** = `TemporalUNet3D` detector + `SimpleNodeTransformer`
  linker — **already ported/rejected** (SOT-2848).
- **Ultrack** (Nature Methods 2025, royerlab): multi-hypothesis contour hierarchy + ILP
  **overlap-maximization** under segmentation uncertainty. The ILP/Gurobi core is
  **non-portable**, but its **temporal-consistency selection principle** is portable to
  our numpy/scipy point-detection pipeline.

Sources: competition page · `royerlab/kaggle-cell-tracking-competition` ·
Ultrack (`nature.com/articles/s41592-025-02778-0`, `biorxiv 2024.09.02.610652v1`,
`arXiv:2308.04526`) · `royerlab/ultrack`.

## Selected axis — port Ultrack's temporal-consistency principle (2 children)

Both are portable numpy/scipy ports, **default-off**, revert-on-reject, exec-compat,
leak-free 4-family LOFO same-seed A/B with a **per-dataset non-regression gate**, and
**must not run any Kaggle submission**.

### Child A (linking-side) — bidirectional motion-consistency link gate
Layer on the **confirmed** SOT-2864 motion linker (`motion_model_link=True`,
sigma=15/gain=1.0/gate_on_prediction=True, CV 0.6760). Add a forward↔backward
mutual-agreement gate (Ultrack overlap surrogate for point detections): a link `t→t+1`
is accepted/cheapened only when the source's forward motion-predicted position and the
target's backward motion-predicted position mutually agree within tolerance; disagreeing
(inconsistent, likely-FP) links are penalized/dropped. New knob `link_consistency_gate`
(default-off), tested vs **both** the frozen champion (0.6649) **and** the motion
reference (0.6760). Distinct from SOT-2871 windowed-association (running-avg velocity,
REJECTED) and SOT-2870 learned edge-gate (REJECTED). File: `src/biohub_tracking/link.py`.

### Child B (detection-side) — Ultrack multi-hypothesis selection by temporal support
Don't commit to a single detection threshold. Generate DoG candidates at a small
threshold-hypothesis ladder, then **select** the final disjoint set by **cross-frame
temporal support** (keep a candidate iff it links into a motion-consistent track;
resolve overlaps by *track support*, not single-frame response). New knob
`detect_hypothesis_select` (default-off). Distinct from SOT-2774 multiscale-DoG (scale
fusion at ONE operating point, REJECTED), SOT-2873 recall-recovery (unconditional
sub-threshold add, REJECTED), and the champion's post-hoc single-threshold short-track
prune. Files: `src/biohub_tracking/detect.py` (+ pipeline coupling for temporal support).

## Children (registered `Todo`, directive `workers: solo=claude:opus, handoff=off`)
- **A** — Ultrack bidirectional motion-consistency link gate (linking).
- **B** — Ultrack multi-hypothesis detection selection by temporal support (detection).

Parent resume run aggregates both, decides submission under the improvement-gate +
CV↔public divergence doctrine, and records the `## 申し送り`.
