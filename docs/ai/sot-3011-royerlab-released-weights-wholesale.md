# SOT-3011 — Official royerlab learned pipeline adopted WHOLESALE via released pretrained weights (offline detect+link substrate)

**Cycle:** SOT-3010 biohub-claude 改善サイクル cycle-8, **direction 1 (role A' wholesale)**, explore-first.
**Verdict:** **REJECT (non-promotable)** — but a **positive mechanistic finding**. No Kaggle submission (child).
Champion `detect-link-dog-v4-shorttrack-motion-gain1` (micro_adj **0.6760**) byte-frozen & untouched.

## Hypothesis / new grounds
The three prior learned-detector/linker rejects trained **from scratch** on the sparse GT
(SOT-2828 numpy re-rank scorer, SOT-2848 fg-MSE, SOT-2863 nnPU/Cellsparse, SOT-2993 masked-sparse UNet3D
`micro_adj 0.6217`, SOT-2841/2870/2994 learned linking). SOT-2992 handoff #1 argued the head-room is the
**detection substrate**, and the SOT-2993 reject was a *from-scratch / under-convergence* failure rather
than a refutation of learned detection (top LB ~0.958 is entirely learned). **New evidence:** the
organiser released **converged public weights** via an inference notebook. Adopting the released
`TemporalUNet3D` detection + `SimpleNodeTransformer` linking **wholesale** (weights bundled offline, *no
training*) measures learned detect+link at its **real operating point**.

## Sources (fetched & read)
- Official baseline repo: https://github.com/royerlab/kaggle-cell-tracking-competition
  (`src/tracking_cellmot/models/{temporal_unet,simple_node_transformer}.py`,
  `scripts/{predict,train}_unet_transformer.py`). Deps torch + tracksdata + zarr + polars + scipy.
- Released weights dataset: `thibautgoldsborough/cellmot-baseline-artifacts`
  (`weights/unet_transformer/split_0/edge_predictor_best.pth` + `config.json` + **62 offline wheels**).
- Released-weights inference notebook: `thibautgoldsborough/unet-baseline-inference-submission`
  (`enable_internet=false`, `enable_gpu=true`; `det_threshold=0.99`, `use_ilp=true`).

## Method (same-seed leak-free A/B, detection-only-vs-full swap)
Both arms scored by the **identical** leak-free harness (`biohub_tracking.eval.cv`: 4-family LOFO,
re-anchored micro_adj edge-Jaccard, SOT-2817/2903). Champion arm = classical DoG+NMS via
`evaluate_cv()` (reproduced **0.676 byte-exact** — harness integrity confirmed). Learned arm reuses the
**official inference code verbatim** (`predict_unet_transformer.predict_video` + `load_model`) on our 4
CV videos, converting each per-video graph to our `TrackingGraph` and scoring with the same
`score_family`/`aggregate`. Runner: `experiments/sot3011/run_ab.py`. Two linkers measured: greedy
(max_parents=1/max_children=2) and the notebook's ILP (tracksdata `ILPSolver`, `edge_weight=-1`,
appear/disappear=0.1, division=1.0 — solved by **SCIP** offline; Gurobi license not required).

Effective config: `unet_out_channels=32`, `unet_layers=[32,64,128]`, `downsample=[1,4,4]`,
`window_size=2`, det TTA on, edge softmax, edge_threshold 0.5, det_threshold 0.99.

## Result — CV table (adjusted edge-Jaccard; **primary KPI = leak-free CV**)

| family (lineage) | champion | learned greedy | Δ | learned **ILP** | Δ |
|---|---|---|---|---|---|
| 44b6_0113de3b (44b6) | **0.9078** | 0.8641 | −0.0437 | 0.8838 | −0.0240 |
| 44b6_0b24845f (44b6) | **0.6938** | 0.5825 | −0.1113 | 0.6262 | −0.0676 |
| 6bba_05b6850b (6bba) | 0.5748 | 0.7874 | **+0.2126** | 0.8368 | **+0.2620** |
| 6bba_05db0fb1 (6bba) | 0.7477 | 0.7630 | +0.0153 | 0.7929 | +0.0452 |
| **micro_adj** | **0.6760** | 0.7705 | **+0.0945** | **0.8081** | **+0.1321** |
| macro | 0.7310 | 0.7493 | +0.0183 | **0.7849** | +0.0539 |
| lineage-macro | 0.7351 | 0.7440 | +0.0089 | **0.7796** | +0.0445 |

- **Decision = REJECT (both linkers)**: micro/macro/lineage-macro **all rise** (so the gain is *not* a
  6bba-mix artifact — the learned pipeline is genuinely stronger in aggregate), **but the mandatory 4/4
  per-dataset non-regression gate FAILS** — both **sparse 44b6** families regress (greedy −0.044/−0.111;
  ILP −0.024/−0.068). The learned detector over-detects where classical DoG is precise on the sparse-
  annotation 44b6 lineage. `division_jaccard=0.0` (max_children=2 but no GT-scored divisions here).

## LEAK CAVEAT (why this is an upper bound, not a promotable number)
The released `split_0` weights were trained on the organiser's labelled train videos, which **include
these 4 CV families** (split membership not independently confirmable; treat as contaminated). So the
learned arm's CV is an **optimistic, train-contaminated upper bound**, not a leak-free estimate. The true
leak-free version — retraining with each family held out — is exactly SOT-2993, which produced **0.6217**.
The **0.62 → 0.77–0.81 gap** is precisely the train-contamination + convergence gap. A *promotable*
learned candidate therefore needs **leak-free retraining to convergence** (SOT-2993's masked-sparse setup
trained longer / with the released architecture & schedule), not the released weights as-is.

## exec-compat (offline Kaggle kernel) — **PORTABLE** (`experiments/sot3011/exec_compat.json`)
Reproduced the organiser's offline path locally: `pip install --no-index --find-links <62 bundled wheels>
tracksdata "zarr>=3.0.10" pyscipopt` (no internet), torch from the GPU base image, `weights_only=True`
attach→load (136 tensors)→forward, tracksdata graph build, and **SCIP** ILP solve — the **full 4-family
GPU A/B ran entirely offline**. This positively refutes the champion `description`'s premise that a
GPU/learned/weights pipeline "cannot run under the submission kernel": it **can** (design consistent with
SOT-2847). ILP needs pyscipopt/SCIP only (Gurobi optional).

## Conclusion & next axis
Converged **learned detection is the real head-room** (first candidate to clear the classical aggregate
ceiling: ILP macro 0.785 vs 0.731), confirming SOT-2992 handoff #1 and re-framing the SOT-2993 reject as
under-convergence, **not** a refutation of learned detection. Two blockers to promotion remain: (1) the
**sparse-lineage (44b6) per-family regression** — a learned-detector operating point / calibration that
does not over-detect on sparse-annotation videos (per-lineage det threshold, or a classical-DoG fallback
on 44b6); (2) **leak-free retraining** so the number is trustworthy. Champion stays byte-frozen; this arm
ships as **default-off evidence** (`experiments/sot3011/`, external sources gitignored). Promotion is the
parent's two-signal call (no submission here).
