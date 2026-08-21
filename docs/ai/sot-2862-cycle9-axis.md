# SOT-2862 — Cycle-9 axis selection (stuck-detection → external-knowledge port)

**Parent:** SOT-2862 (`[biohub-claude] Kaggle順位向上サイクル第6次` — ledger cycle counter = 9).
**Banner:** stuck-detection (実LB順位 flat, 圏外/top20). Local tuning is forbidden this cycle;
the mandate is to **research past Kaggle top solutions and port ≥1 effective, portable method**
(escalation ladder step-6 external knowledge, pulled forward).

Champion `detect-link-dog-v4-shorttrack` is **byte-frozen** this run (config sha
`42064648…`, leak-free CV micro-adj **0.6649**, best public **0.624**, leader **0.957**). No
Kaggle submission on the initial decomposition run.

## What the research found (web, sources below)

The competition is the **royerlab / CZ Biohub** cell-tracking-during-development challenge on 3D+time
light-sheet embryos. Ground truth is **sparse** `.geff` tracking GT (nodes `(t,z,y,x)`, edges link
consecutive-frame cells, divisions = one node → two at `t+1`); scored on edge + division TP/FP/FN with
micro-Jaccard. Structural fact that reframes our ladder: the **official baseline
(`TemporalUNet3D` detector + `SimpleNodeTransformer` linker) already masks its loss to annotated
edges only** — so the sparse-GT difficulty is a **detector-side** problem, *not* a linker problem.
This is consistent with SOT-2847's verdict (0.62→0.89 is a detector-quality gap, not an
infra/feasibility gap) and with SOT-2848 (a *from-scratch-trained* learned detector was degenerate on
sparse GT).

### Ranked NEW portable axes surfaced (not previously tried here)

1. **nnPU (non-negative Positive-Unlabeled) detection loss** — the direct antidote to SOT-2848's
   rejection cause. Sparse GT means *unlabeled voxels are cells+background mixed*; naive supervised
   loss treats them all as negatives (**PU contamination**) → the exact degeneracy SOT-2848 hit
   (thr 0.5 → all-zero; thr 0.3 → family-variance explosion; no family-robust operating point). nnPU
   estimates the negative risk as `R̂_U(-1) − π·R̂_P(-1)` and **clamps it at 0** when negative, so the
   net stops overfitting the mislabeled "negatives": `R̂ = π·R̂_P(+1) + max(0, R̂_U(-1) − π·R̂_P(-1))`,
   `π` = expected cell-voxel fraction (estimable from GT node density). ~30-line loss swap, PORTABLE
   (train offline, attach weights offline; GPU allowed). Complement: **Cellsparse ignore-weighting**
   (weight unlabeled-pixel loss ~0.05 instead of 0/1) for the regression head.
2. **ARGUS optical-flow motion-model linking** — FULLY PORTABLE, CPU-only, no weights/training.
   Predicts each cell's next position via dense (Farneback) optical flow, then LAP-matches
   predicted-vs-detected within a gating radius. Mechanistically **distinct** from our rejected static
   gap-closing (SOT-2763, non-consecutive edges dropped) and node-interp gap-recovery (SOT-2849): it
   is a *first-order* linking-cost change (predict **where the cell went**), whereas the champion
   linker is distance-only nearest-neighbour with **no motion model**.
3. **Ultrack multi-hypothesis segmentation + ILP operating-point selection** — mostly portable with an
   open-source solver (HiGHS/CBC via PuLP/python-mip). Picks the detection operating point *per-cell*
   via a global temporal-consistency ILP instead of one global threshold — attacks the recurring
   "no single global operating point for sparse over-counting" wall. Heavier lift; **queued for a
   later cycle**, not this cycle's child.
4. **Trackastra parental-softmax degree-constrained pruning** — pure-logic association pruning
   (≤1 parent / ≤2 children) on any cost matrix. Overlaps the division-FP hazard that rejected
   SOT-2762; **queued as a future axis**, not a child now.

## Selected axis this cycle (→ 2 children)

Port the research's recommended pairing — **one portable learned axis + one portable pure-logic axis**,
attacking the detection and linking halves independently:

- **Child A — nnPU-corrected learned detector** (detector-quality; the primary external-knowledge port).
  Reuses SOT-2848's LOFO training harness + SOT-2847 receptacle; swaps the naive foreground-weighted MSE
  for the nnPU non-negative risk estimator (+ a Cellsparse ignore-weighted variant). Leak-free LOFO CV
  same-seed A/B vs the SOT-2848 naive-loss detector **and** vs the classical champion. Default-off unless
  it clears the promotion gate. **Not a blind retry of SOT-2848** — nnPU is a new method with new
  grounds that targets SOT-2848's *identified root cause* (PU contamination), which every prior
  learned-detector attempt (SOT-2828/2848) ignored.
- **Child B — motion-model-predicted LAP linking** (ARGUS; fully-portable CPU, no weights). Default-off
  `LinkParams` path; A/B on champion classical detection (measurable) on the leak-free CV, per-dataset
  no-regression gate. cv2 portability is verified against the offline kernel; falls back to a
  numpy/scipy constant-velocity motion model if OpenCV is non-attachable.

Both children: no Kaggle submission (submission is the parent resume run's responsibility, gated by the
improvement gate / reserve / spacing / fingerprint); champion byte-frozen unless promoted; screen→confirm
gate; non-promotion ⇒ revert to default-off + record in `docs/`; rejected/CLOSED verdicts require
same-seed A/B evidence (else inconclusive); every evaluated axis appends a JSONL entry to
`docs/ai/experiment_ledger.jsonl`.

## Reserve / carry-over (unchanged, for the parent resume run's LB-probe slot)

- **SOT-2849** node-interpolation gap-recovery (family_mix_sensitive; micro 0.6649→0.6773 on 3/4
  families) — reserve/終盤 LB-probe **#1**.
- **SOT-2840** global-MCF θ=6.5/window=2 candidate artifact — reserve/終盤 LB-probe **#2**.

## Sources

- competition + baseline: https://forum.image.sc/t/biohub-cell-tracking-during-development-kaggle-competition/121671 · https://github.com/royerlab/kaggle-cell-tracking-competition
- nnPU: https://arxiv.org/pdf/1703.00593 · impl https://github.com/kiryor/nnPUlearning/blob/master/pu_loss.py · PU cell detection https://arxiv.org/pdf/2106.15918
- Cellsparse / Sketchpose masked loss: https://www.biorxiv.org/content/10.1101/2023.06.13.544786v1.full · https://www.melba-journal.org/papers/2025:016.html
- ARGUS optical-flow tracker: https://arxiv.org/html/2607.08297
- Trackastra: https://arxiv.org/html/2405.15700v1
- Ultrack: https://arxiv.org/pdf/2308.04526 · https://github.com/royerlab/ultrack
