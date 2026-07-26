# biohub-claude

Working repository for the Kaggle competition
**[Biohub — Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)**
(system lineage: `claude`).

The task: from 3D + time light-sheet microscopy videos of a developing zebrafish
embryo, detect cells and reconstruct their **tracking graph** — including cell
**divisions**. This repo bootstraps the shared foundation the later stages build
on: a **data loader** for the tracking-graph I/O formats and a **local
evaluator** that reproduces the competition's Edge/Division Jaccard score so
candidates can be compared offline.

> **The data schema below was confirmed against the real Kaggle data** (this
> environment is Kaggle-authenticated). One real training annotation
> (`44b6_0113de3b.geff`, 52 nodes / 50 edges) was downloaded and used to verify
> the loader, and the metric was validated against the organiser's own reference
> implementation (see [Validation](#validation)).

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt
```

Dependencies are deliberately light — **numpy, scipy, zarr** only — so the loader
and evaluator run unchanged inside a Kaggle submission kernel.

## Layout

```
src/biohub_tracking/
  graph.py          TrackingGraph: nodes (t, z, y, x) + directed edges
  io.py             geff (ground truth) + submission-CSV read/write
  matching.py       per-timepoint ≤7 µm optimal bipartite node matching
  eval/
    edge_metric.py      Edge Jaccard TP/FP/FN
    division_metric.py  Division Jaccard TP/FP/FN (local-window topology)
    score.py            adjusted edge Jaccard + combined score, micro-averaged
  evaluate.py       CLI: score a submission CSV against a GT geff directory
tests/              unit tests incl. golden cases from the official reference
data/               competition data (gitignored — not redistributable)
```

## Data schema

### Ground truth — `*.geff` (Graph Exchange File Format)

Each training dataset is a `<name>.geff` directory (a **zarr v3** group,
`geff_version` 1.1, `directed: true`):

| path | meaning |
| --- | --- |
| `zarr.json` → `attributes.geff` | metadata: `axes` (scales), `extra.estimated_number_of_nodes` |
| `nodes/ids` | `uint64` node ids, shape `(N,)` |
| `nodes/props/t/values` | `int64` timepoint per node |
| `nodes/props/{z,y,x}/values` | `int64` voxel centroid coordinates per node |
| `edges/ids` | `uint64` `(source_id, target_id)` pairs, shape `(E, 2)` |

- **Node** = one cell detection: a timepoint `t` and a 3D centroid `(z, y, x)` in
  **voxel** units.
- **Edge** = a directed link from a cell at `t` to the same cell (or, at a
  division, one of its two daughters) at `t + 1`.
- A **division** is a node with exactly two outgoing edges.
- Ground truth is **sparse**: only a subset of cells is annotated (the sample
  dataset has 52 nodes vs an estimated ~25 755 true nodes).

**Physical scale (anisotropic voxels), read from `geff.axes`:**
`(z, y, x) = (1.625, 0.40625, 0.40625)` µm per voxel. All distances in the metric
are computed in microns, so this scale is applied to centroid coordinates before
matching.

### Predictions — submission CSV

A single flat CSV; each row is a node **or** an edge, distinguished by
`row_type`:

```
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
0,44b6_0113de3b,node,1,0,32,128,128,-1,-1
3,44b6_0113de3b,edge,-1,-1,-1,-1,-1,1,2
```

| `row_type` | populated columns | unused columns (set to `-1`) |
| --- | --- | --- |
| `node` | `node_id, t, z, y, x` | `source_id, target_id` |
| `edge` | `source_id, target_id` | `node_id, t, z, y, x` |

`id` is a 0-based row index; `dataset` groups rows by video (matches the geff
stem). Edge rows reference `node_id`s within the same `dataset`.

The test images ship as OME-NGFF `*.zarr` (dtype `uint16`, shape `(T, Z, Y, X)`);
reading those pixels is the job of the detection stage (SOT-1983), not this
loader.

## Evaluation metric

The leaderboard score follows the organiser's specification
(`royerlab/kaggle-cell-tracking-competition/metrics.md`):

1. **Node matching** — predicted nodes are matched to GT nodes **per timepoint**
   by centroid distance, up to **7 µm**, via an **optimal (minimum-cost)
   bipartite assignment** (one-to-one). Distances use the physical voxel scale.
2. **Edge Jaccard** = `TP / (TP + FP + FN)`:
   - **TP** — a predicted edge whose both endpoints match GT nodes joined by a GT
     edge.
   - **FN** — a GT edge with no such match.
   - **FP** — a non-TP predicted edge that is *evaluable* against the sparse GT
     (its source matches a GT node with an outgoing edge, or its target matches a
     GT node with an incoming edge). Predictions outside the annotated region are
     ignored, not penalised here.
   - **Adjusted** — scaled by a node-count penalty
     `J_adj = max(0, J·(1 − 0.1·(N_pred − N_true)/N_true))`, where `N_true` is the
     geff `estimated_number_of_nodes`. (There is no upper clamp, so predicting
     *fewer* nodes than `N_true` can push `J_adj` above 1 — this is faithful to
     the reference.)
3. **Division Jaccard** = `TP / (TP + FP + FN)` over division events, evaluated in
   a local `grandparent → parent → children → grandchildren` window that tolerates
   the fork being one timepoint early/late.
4. **Combined score** = `adjusted_edge_jaccard + 0.1 · division_jaccard`,
   **micro-averaged** across datasets (TP/FP/FN summed before the ratio; the
   adjusted edge term is size-weighted by `TP+FP+FN`).

## Usage

```python
from biohub_tracking.io import load_geff, load_submission_csv
from biohub_tracking.eval import evaluate

gt   = load_geff("data/train/44b6_0113de3b.geff")
pred = load_submission_csv("submission.csv")["44b6_0113de3b"]
result = evaluate(pred, gt, scale=(1.625, 0.40625, 0.40625))
print(result)   # EvaluationResult(edge_tp=..., division_tp=..., ...)
```

Score a whole submission against a directory of GT geffs:

```bash
biohub-evaluate --pred submission.csv --gt-dir data/train
# or: python -m biohub_tracking.evaluate --pred submission.csv --gt-dir data/train
```

## Getting the data

Requires a Kaggle account joined to the competition:

```bash
kaggle competitions download -c biohub-cell-tracking-during-development -p data/
# training annotations are the train/*.geff directories; test images are test/*.zarr
```

## Validation

Correctness is pinned by `tests/test_sandbox_golden.py`: the eight hand-crafted
graph cases (perfect / missed / delayed / spurious / cross-component divisions,
plus the `hack2` exploit topology) with their frozen expected TP/FP/FN counts are
ported from the organiser's reference test suite. This reimplementation
reproduces **all** of them, so the local score tracks the real Kaggle metric. See
[`NOTICE.md`](NOTICE.md) for attribution.

```bash
pytest -q
```

## Next stages

- **SOT-1983** — detection + linking baseline champion (reads `test/*.zarr`,
  emits a tracking graph), gated by this local score.
- **SOT-1984** — exec-compatible submission packaging + real Kaggle submission.
