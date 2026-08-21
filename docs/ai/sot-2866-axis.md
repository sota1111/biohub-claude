# SOT-2866 — cycle axis selection (stuck-detection → external-knowledge port, ladder step-6)

**Run type: initial (decomposition).** No Kaggle submission this run. Champion
`detect-link-dog-v4-shorttrack` `config.json` sha256 `42064648e612183e761bf9d40b70d3e8a2497453a878f1a44f5b52e410e01bdd`
byte-frozen (== last-submitted public 0.509 / CV micro-adj 0.6649). biohub-claude
deadline 2026-09-29 → **mode=improve** (not converge); daily_reserve=1, min_interval_min=180.

## Stuck signal & mandate

The cron banner reports the public LB rank is flat (圏外/out-of-top-20, 3 obs). Local
tuning is banned this cycle; the mandated axis is **research + port this competition's
top solutions** (escalation ladder step-6, external knowledge).

## External research (web, 2026-08-21)

- **No winner / medal "Solution" write-ups exist yet** — the competition is still
  active (launched 2026-06-29); public notebooks are baseline-derivative (classical
  baseline, scoring explainer, EDA) only. So "port a top solution" resolves to porting
  the **official royerlab baseline's learned-linker family (Trackastra)**, adapted to
  our 3D + sparse annotated-edge GT.
  - comp: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development
  - baseline repo: https://github.com/royerlab/kaggle-cell-tracking-competition
  - image.sc announce: https://forum.image.sc/t/biohub-cell-tracking-during-development-kaggle-competition/121671
  - classical baseline NB: https://www.kaggle.com/code/xiaoleilian/biohub-cell-tracking-classical-baseline
  - scoring explained NB: https://www.kaggle.com/code/harshitsama/biohub-scoring-data-fully-explained

- **Metric (load-bearing).** Custom, NOT CTC TRA/AOGM
  (https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md):
  predicted nodes matched to GT nodes by bipartite assignment within **7 µm**; an
  **edge is TP only when BOTH endpoints match GT nodes joined by a GT edge** (FP if an
  endpoint matches a GT node linked to a *different* node); **unmatched predicted nodes
  incur NO node FP** (sparse GT); `score = adjusted_edge_jaccard + 0.1·division_jaccard`,
  micro-averaged, with an over-prediction penalty `jaccard·(1 − 0.1·(T_pred−T_true)/T_true)`.
  ⇒ **edge linking dominates division 10:1**, and gains come from **recovering TP edges
  (FN-edge recovery)** without tripping the over-prediction penalty — precisely the
  lever SOT-2864 moved (+0.0111 via gate expansion).

- **Trackastra** (arXiv 2405.15700 / github.com/weigertlab/trackastra): shallow
  features only (center coords w/ Fourier PE, **mean intensity, area, inertia tensor**,
  timepoint) → transformer over a ~6-frame sliding window, distance-masked all-to-all
  attention, association logits, **division-aware "parental softmax"** (block-wise
  parent-assoc sum ≤ 1). CPU-runnable (position/shape variant, not SAM2); pretrained
  weights attachable offline as a Kaggle dataset. **Caveat:** pretrained models are 2D
  + dense-mask trained; our data is 3D + sparse point GT → adopt its **feature set and
  parental-softmax mechanism, NOT its weights**.

- **ultrack** (arXiv 2308.04526): global ILP min-cost-flow association; but Gurobi
  (license) / CBC (heavy) — not offline-kernel friendly. Not selected.

## Prior-cycle evidence that constrains the axis

- **SOT-2841 (learned edge re-ranker) = REJECTED.** A logistic edge classifier over
  `[dist_scaled, app_cos, src_rivals, dst_rivals, succ_rank, pred_rank, succ_margin]`
  was genuinely discriminative (LOFO pos-vs-neg gap **0.42**) but produced **no CV
  gain**: re-ranking within the champion's **fixed** raw-distance feasible set is
  saturated (near-distance ≈ being-the-GT-edge). It lacked **motion** and **shape/
  intensity** features and only re-ranked (feasibility gate stayed on raw distance).
- **SOT-2864 (motion-model linking) = CV-promotable/+0.0111, all-4-families
  non-regressing, NOT submitted.** The gain came from `motion_gate_on_prediction=True`
  — admitting fast, motion-consistent successors the **raw distance gate drops**
  (FN-edge recovery), NOT from re-ranking. **→ headroom lives in the feasibility gate,
  not the re-rank.**
- SOT-2848/2863: learned per-voxel detector ladder exhausted on sparse GT (naive
  degenerate; nnPU/Cellsparse recover but no family-invariant operating point).

## Selected axis & decomposition (3 children, all target the confirmed edge lever)

All children: leak-free 4-family LOFO CV A/B on champion classical detection,
per-dataset no-regression gate, default-off, champion byte-frozen, **no Kaggle
submission** (parent resume aggregates & decides). Directive
`workers: solo=claude:opus, handoff=off`.

1. **Learned edge-cost linker with motion + shape features that EXPANDS feasibility
   (FN-edge recovery)** — the #1 handoff axis (learned linking) done with new grounds:
   add the motion-predicted-position residual (SOT-2864, the +0.0111 lever) and
   Trackastra shape/intensity ratios (mean-intensity ratio, size/inertia ratio) — the
   features SOT-2841 lacked — and use the learned score to **admit** motion/shape-
   consistent successors the raw distance gate drops (not merely re-rank the fixed set).
   Trained on the sparse **annotated-edge** GT (the exact label the metric scores).
   CPU numpy/scipy + embedded coefficients. New grounds vs SOT-2841 (re-rank saturated,
   no motion/shape features) and SOT-2864 (hand-tuned gate, not learned).

2. **Trackastra windowed association port (features + parental-softmax + short sliding
   window), portable (numpy/scipy min-cost-flow, no torch/weights)** — escalate t→t+1
   greedy to **windowed global association** over a ~2–3-frame window using the
   motion/learned edge cost, with Trackastra's **parental-softmax** division constraint
   (parent-assoc sum ≤ 1). Directly ports the top-solution *mechanism* the stuck banner
   asks for, without the 2D-dense-pretrained / torch dependency. New grounds vs static
   gap-closing (SOT-2763, non-continuous edge metric) and node-interp gap-recovery
   (SOT-2849, family-mix-sensitive): a learned/motion-scored windowed association, not a
   geometric interpolation.

3. **Recall-oriented detection to recover FN-edge endpoints, exploiting the
   no-FP-for-unmatched-predictions metric** — an edge TP needs BOTH endpoints detected
   within 7 µm; the metric charges **no node FP** for unmatched predictions (only the
   mild global over-prediction penalty). Isolate **GT-node recall @7 µm** (not the
   aggregate score) as the objective and produce the recall-vs-over-prediction-penalty
   tradeoff; promote only if per-dataset non-regressing. New grounds vs the exhausted
   per-voxel operating-point ladder (SOT-2789): recall-as-objective under the metric's
   no-FP property, not a family-invariant per-voxel magnitude. (A null result is a clean
   inconclusive/reject-with-evidence, not a false promotion.)

## Not selected / deferred

- CV↔LB transfer-coefficient measurement (standing #2 axis) — requires spending an LB
  slot; children cannot submit. Deferred to the parent resume run / endgame reserve.
- Full Trackastra pretrained-weight attach — blocked by 3D+sparse vs 2D-dense mismatch.
- Dedicated division-aware linking child — division is only 0.1× the score (SOT-2762
  division-linking already rejected on fork-FP loss); folded into child 2's parental
  softmax instead.

## Standing reserve / final-window LB-probe queue (unchanged, champion NOT flipped)

1. SOT-2864 motion-model linking (+0.0111, all-family non-regressing, best transfer profile)
2. SOT-2849 node-interp gap-recovery (+0.0124 but family-mix-sensitive)
3. SOT-2840 global-MCF θ=6.5 (+0.0022, family_mix_sensitive)
