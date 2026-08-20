# SOT-2846 — Cycle-8 axis: external-knowledge escalation to the learned pipeline

Kaggle 順位向上サイクル (parent SOT-2846). Stuck-detection banner fired (real LB flat, out of
top-20; champion public 0.509 / best 0.624; champion leak-free CV micro-adj 0.6649). Per the
mandate this cycle front-loads escalation-ladder step (6) *external knowledge* + step (5)
*architecture change*, gated by step (3) *generalization-gap diagnosis*.

## What the classical ladder has exhausted (cycles 2–7,台帳)

Every classical **detection** operating-point (single/multi-scale DoG, per-volume quantile
normalization, watershed / density-gated split, Hessian-blobness, local-adaptive MAD, GT-learned
numpy detection scorer) and every classical **linking** re-rank (division-aware, gap-closing LAP,
global min-cost-flow θ-sweep, appearance descriptors, GT-learned edge cost) was rejected under
same-seed leak-free 4-family CV A/B. Shared root cause: the **sparse-GT node-count penalty**
(`adjusted_jaccard = jaccard·(1 − 0.1·(T_pred − T_true)/T_true)`) makes no single *global*
classical operating point simultaneously good on the dense fused `6bba` family and the sparse
`44b6` family — the classical pipeline is saturated at ~0.66 CV / ~0.62 public.

## External knowledge — the frontier score-source (with sources)

- **royerlab official baseline** `github.com/royerlab/kaggle-cell-tracking-competition`: end-to-end
  **learned** pipeline — (1) detection = a 3D U-Net with temporal attention (`TemporalUNet3D`)
  producing a per-voxel detection map, centres via local-max suppression; (2) linking = a
  cross-attention transformer (`SimpleNodeTransformer`) scoring every (t, t+1) node pair;
  (3) **sparse supervision** — only GT-covered edges backprop. Metric (authoritative, `metrics.md`):
  node matching by optimal ≤7µm bipartite assignment; adjusted edge Jaccard (node-count penalty
  a=0.1) + 0.1·division Jaccard, micro-averaged.
- **Public UNet baseline *inference* submission notebook** (thibautgoldsborough) — trains via the
  royerlab `train_unet_transformer.py` and **submits a learned model**, proving a trained model can
  produce a valid submission in *this* competition.
- **`pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery`** — public ≈**0.890** (vs LB
  leader 0.957): learned graph **+ gap recovery** post-processing. The gap-recovery step is the
  differentiator over the bare baseline.
- Public **classical** baseline (`xiaoleilian/biohub-cell-tracking-classical-baseline`) and a
  frontier lineage tracker (public 0.913, GPU pretrained UNet+ILP) corroborate: the ~0.62→0.89+
  gap is a **detector-quality (learned vs classical)** gap, not a linking gap.

## The generalization-gap layer (ladder step 3) — the codified, UNVERIFIED constraint

`champion/config.json` description asserts the frontier "cannot run under this repo's
numpy/scipy/zarr, CPU, no-internet, no-weights kernel." **This assumption has never been tested**
and is contradicted by the existence of public notebooks that submit learned models. If the
competition kernel actually permits an **attached weights Dataset + GPU inference offline** (the
standard Kaggle offline-model pattern), the assumption is the single false premise keeping this
project classical and capped at 0.62 — removing it unlocks the entire frontier. Resolving this is
therefore the first child and the pivot of the cycle.

Container feasibility confirmed locally: `torch 2.5.1+cu121`, CUDA available (RTX 3080 Ti),
`data/train/` present (1.8 GB, the 4 CV families with paired `.geff` GT) → a U-Net can be trained
and evaluated here; GPU is explicitly allowed for children.

## Decomposition (3 children, sequential; none may submit)

1. **A — submission-constraint re-verification + learned-inference receiver** (no dep): empirically
   determine whether the biohub kernel allows attached-weights + GPU torch inference offline (verify
   against the public learned-inference notebooks / competition kernel type); wire a default-off
   learned-detection inference path + weights-attach smoke test into the pipeline & `build_kernel`,
   champion byte-frozen. Records the verdict in the ledger. Unblocks or honestly blocks 2/3.
2. **B — port the learned 3D U-Net detector** (blockedBy A): train/adopt `TemporalUNet3D`
   detection locally, feed existing linking, same-seed A/B vs the 0.6649 classical champion on the
   leak-free 4-family CV with the per-dataset no-regression gate. Default-off flag; byte-frozen
   unless promoted.
3. **C — learned linking (`SimpleNodeTransformer`) or gap-recovery post-processing** (blockedBy B):
   on top of learned detection, port the transformer node-pair linker or the pilkwang gap-recovery
   step; A/B on CV.

Selection/submission discipline unchanged: promote only on a leak-free-CV improvement beyond the
noise band with no per-dataset regression; children never submit; exec-compat + champion byte-check
gates apply on promotion. Initial run registers children and waits at In Review (no submission).
