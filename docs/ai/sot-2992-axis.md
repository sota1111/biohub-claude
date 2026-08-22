# SOT-2992 — Cycle-7 explore-first axis selection (biohub-claude)

**Mode:** stuck-detection (real-LB flat, out-of-top20 ×4) + `explore_first=true`.
Champion micro-diffs are forbidden this cycle. Instead: a portfolio of structurally-independent
directions, with the **mandated main axis = port external knowledge** (this competition's top
solutions + its own public notebooks / official baseline).

## State going in

- Champion `detect-link-dog-v4-shorttrack-motion-gain1`: classical CPU-only DoG detection +
  short-track pruning + ARGUS motion-model LAP linking. Leak-free CV micro_adj **0.6760**, public
  **0.626**. LB leader **0.958** (gap 0.332), out-of-top20 ×4 (flat).
- Escalation ladder step 1–6 walked across cycles 3–6; every classical variant non-promoted.
  Confirmed wall: **density-mix** (SOT-2921 — family 6bba contains both the sparsest and densest
  regimes, so observable covariates crosscut the true difficulty boundary; a single global operating
  point cannot win both). Rejected this route: single global op-point, hard/soft regime conditioning,
  independent classical foundation, division overlay (×4), CV holdout re-granularity.

## External research (mandated — web search)

- **Official baseline `royerlab/kaggle-cell-tracking-competition`**
  (https://github.com/royerlab/kaggle-cell-tracking-competition): **`TemporalUNet3D`** — 3D U-Net with
  temporal attention producing per-voxel features + a single-channel detection map; centers via
  local-max suppression. Linking: **`SimpleNodeTransformer`** — cross-attention transformer scoring
  (t, t+1) node pairs from pooled node features. **Masked sparse supervision**: "only edges with
  ground truth are used for backpropagation — background detections and unannotated cells are ignored."
  Metric: `evaluate.py` (edge & division TP/FP/FN + micro-averaged Jaccard; `metrics.md`). Data:
  OME-Zarr `(T,Z,Y,X)` @ `1.625/0.40625/0.40625` µm; GEFF track graphs (sparse nodes, division = one
  parent → two child edges). GPU: U-Net train + transformer + feature extraction. CPU-portable: graph
  matching, `geffs_to_csv.py`/`csv_to_geffs.py`, `evaluate.py`.
- Competition: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development — zebrafish
  embryo 3D videos; **sparse GT** (only a subset of cells annotated per video).
- ISBI Cell Tracking Challenge precedent: U-Net (2015 winner) and KIT-Sch-GE two-decoder U-Net
  dominate learned detection/segmentation — corroborates that this class of problem is a
  learned-detector game, not a classical-filter game.

**Key realization:** biohub is a **learned-detector game**; the classical ceiling (~0.676 CV / 0.626
public) is real and saturated. The frontier substrate is the official learned baseline. GPU is
permitted this cycle and we **train our own weights** (portable — we own and ship them), so the
"GPU-weights-non-portable" ceiling that blocked *porting pretrained* weights does not apply to
*self-training*. The prior learned attempt SOT-2828 died on **PU contamination** (sparse GT →
unannotated real cells treated as negatives); the official **masked loss** is the exact new mechanism
that defeats it — this is the *new evidence* that legitimately reopens the learned-detector axis.

## Portfolio (4 structurally-independent children — no submission this run)

| # | Role | Direction | New-evidence vs prior reject |
| - | ---- | --------- | ---------------------------- |
| 1 | A′ / arch | **Self-trained 3D U-Net detection** (TemporalUNet3D port, masked sparse loss, own weights, offline artifact) — replaces DoG wholesale; classical hedge kept | Masked loss defeats SOT-2828 PU contamination |
| 2 | A′ / linking | **Learned cross-attention edge linking** (SimpleNodeTransformer port) — learned edge score replaces motion-LAP cost; runnable on classical features if C1 unmerged | Learns edge score vs switching a fixed op-point (≠ 2922/2923/2931) |
| 3 | B / oracle | **Port official `evaluate.py` metric as the CV scorer** + re-quantify CV↔public transfer-trust under the exact competition metric | Ports the official scorer (fidelity fix), not holdout granularity (≠ SOT-2929) |
| 4 | C / reformulation | **Semi-supervised dense pseudo-label self-training** — teacher densifies labels on unannotated cells to relieve the node-count penalty capping mixed-density 6bba | Densifies supervision (root cause) vs op-point switching; teacher-densified ≠ SOT-2828 sparse-GT direct |

## Guardrails (inherited)

- **Primary KPI = leak-free CV**; public LB is secondary sanity. On divergence, believe CV
  (rogii: public↔private τ=−1.00 possible). **public-best selection is forbidden.**
- Promotion = **two-signal gate** (CV↑ past noise, 4/4 per-dataset non-regression, AND public
  non-contradicting). `cv_representative=false` → adoption is a bet → keep the classical champion as
  hedge, never all-in.
- Every child: screen→confirm gate, revert+docs on non-promotion, exec-compat (offline / no-internet
  / no-`__file__`) on promotion. **Children must not submit to Kaggle.** Submission (if any) is the
  parent-resume run's decision under reserve/spacing/cap via
  `scripts/ai/kaggle_targets_submit.sh`.
- Rejected axes are not retried without the new evidence recorded above and in
  `docs/ai/experiment_ledger.jsonl` (cycle 7 entries).
