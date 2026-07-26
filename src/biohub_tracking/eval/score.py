"""Per-sample evaluation and run-level aggregation.

Combines the edge and division counts into the competition score:

* **Adjusted edge Jaccard** — the edge Jaccard scaled by a penalty on the total
  number of predicted nodes, ``J_adj = max(0, J·(1 − α·(N_pred − N_true)/N_true))``
  with ``α = 0.1``. ``N_true`` is a coarse per-dataset estimate of the true node
  count (the geff ``estimated_number_of_nodes`` metadata).
* **Division Jaccard** — micro-averaged (TP/FP/FN summed across datasets).
* **Combined score** — ``adjusted_edge_jaccard + w · division_jaccard`` with
  ``w = 0.1``.

Aggregation is **micro-averaging**: per-sample TP/FP/FN are summed across the
whole split before the Jaccard is computed. The reported adjusted edge Jaccard is
the per-sample adjusted Jaccard weight-averaged by sample size
``w_i = TP_i + FP_i + FN_i``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from ..graph import TrackingGraph
from ..matching import DEFAULT_MAX_DISTANCE, match_nodes
from .division_metric import division_counts
from .edge_metric import edge_counts

ADJUSTMENT_ALPHA: float = 0.1
SCORE_DIVISION_WEIGHT: float = 0.1


class EvaluationResult(NamedTuple):
    edge_tp: int
    edge_fp: int
    edge_fn: int
    division_tp: int
    division_fp: int
    division_fn: int
    num_pred_nodes: int


class DatasetsResult(NamedTuple):
    edge_jaccard: float
    division_jaccard: float
    adj_edge_jaccard: float
    score: float


def _jaccard(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return tp / denom if denom > 0 else float("nan")


def evaluate(
    pred: TrackingGraph,
    gt: TrackingGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> EvaluationResult:
    """Evaluate one predicted graph against one ground-truth graph."""
    matching = match_nodes(pred, gt, scale=scale, max_distance=max_distance)
    ec = edge_counts(pred, gt, matching)
    dc = division_counts(pred, gt, scale=scale, max_distance=max_distance)
    return EvaluationResult(
        edge_tp=ec.tp,
        edge_fp=ec.fp,
        edge_fn=ec.fn,
        division_tp=dc.tp,
        division_fp=dc.fp,
        division_fn=dc.fn,
        num_pred_nodes=pred.num_nodes,
    )


def adjusted_edge_jaccard(
    edge_jaccard: float, num_pred_nodes: int, n_true: float
) -> float:
    """Scale an edge Jaccard by the predicted-node-count penalty."""
    if math.isnan(edge_jaccard) or n_true is None or n_true <= 0 or math.isnan(n_true):
        return float("nan")
    total_node_ratio = (num_pred_nodes - n_true) / n_true
    return max(0.0, edge_jaccard * (1 - ADJUSTMENT_ALPHA * total_node_ratio))


def evaluate_datasets(
    graph_pairs: list[tuple[TrackingGraph, TrackingGraph]],
    n_true: list[float] | None = None,
    scale: tuple[float, ...] | None = None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> DatasetsResult:
    """Evaluate many (pred, gt) pairs and return the micro-averaged score.

    Parameters
    ----------
    graph_pairs
        ``(pred_graph, gt_graph)`` pairs.
    n_true
        Per-pair coarse true-node-count estimates for the adjusted edge Jaccard.
        ``None`` (or NaN entries) drops the node-count penalty for that pair and
        falls back to the raw edge Jaccard weight-averaged the same way.
    """
    results = [
        evaluate(pred, gt, scale=scale, max_distance=max_distance)
        for pred, gt in graph_pairs
    ]
    n_true_list: list[float] = (
        [float("nan")] * len(results) if n_true is None else list(n_true)
    )

    edge_tp = sum(r.edge_tp for r in results)
    edge_fp = sum(r.edge_fp for r in results)
    edge_fn = sum(r.edge_fn for r in results)
    div_tp = sum(r.division_tp for r in results)
    div_fp = sum(r.division_fp for r in results)
    div_fn = sum(r.division_fn for r in results)

    edge_jaccard = _jaccard(edge_tp, edge_fp, edge_fn)

    # Adjusted edge Jaccard: per-sample adjusted Jaccard, weight-averaged by
    # sample size w_i = tp + fp + fn.
    weighted_sum = 0.0
    total_w = 0.0
    for r, nt in zip(results, n_true_list):
        w = r.edge_tp + r.edge_fp + r.edge_fn
        j = _jaccard(r.edge_tp, r.edge_fp, r.edge_fn)
        adj = adjusted_edge_jaccard(j, r.num_pred_nodes, nt)
        # Fall back to the raw per-sample Jaccard when no node estimate is given.
        if math.isnan(adj):
            adj = j
        if w > 0 and not math.isnan(adj):
            weighted_sum += w * adj
            total_w += w
    adj_edge_jaccard = weighted_sum / total_w if total_w > 0 else float("nan")

    has_divisions = (div_tp + div_fp + div_fn) > 0
    if has_divisions:
        division_jaccard = _jaccard(div_tp, div_fp, div_fn)
        score = adj_edge_jaccard + SCORE_DIVISION_WEIGHT * division_jaccard
    else:
        division_jaccard = float("nan")
        score = adj_edge_jaccard

    return DatasetsResult(
        edge_jaccard=edge_jaccard,
        division_jaccard=division_jaccard,
        adj_edge_jaccard=adj_edge_jaccard,
        score=score,
    )
