"""Adjusted edge Jaccard, division weighting, and micro-averaged aggregation."""

from __future__ import annotations

import math

from biohub_tracking.eval import evaluate, evaluate_datasets
from biohub_tracking.eval.score import (
    SCORE_DIVISION_WEIGHT,
    adjusted_edge_jaccard,
)
from biohub_tracking.graph import TrackingGraph


def _line(n: int, offset: float = 0.0) -> TrackingGraph:
    """A simple linear track of *n* nodes, one per timepoint."""
    nodes = {i: (float(i), 0.0, offset, 0.0) for i in range(n)}
    edges = [(i, i + 1) for i in range(n - 1)]
    return TrackingGraph.from_lists(nodes, edges)


def test_adjusted_jaccard_penalises_overprediction() -> None:
    # Predicting more nodes than the true estimate scales the Jaccard down.
    assert adjusted_edge_jaccard(1.0, num_pred_nodes=110, n_true=100) == 1.0 * (1 - 0.1 * 0.1)
    # Predicting the exact true count leaves the Jaccard unchanged.
    assert adjusted_edge_jaccard(0.8, num_pred_nodes=100, n_true=100) == 0.8
    # Underprediction inflates it (no upper clamp — faithful to the reference).
    assert adjusted_edge_jaccard(1.0, num_pred_nodes=50, n_true=100) > 1.0
    # Floored at zero for extreme overprediction.
    assert adjusted_edge_jaccard(1.0, num_pred_nodes=10_000, n_true=100) == 0.0


def test_adjusted_jaccard_nan_without_estimate() -> None:
    assert math.isnan(adjusted_edge_jaccard(1.0, 100, float("nan")))
    assert math.isnan(adjusted_edge_jaccard(float("nan"), 100, 100))


def test_perfect_prediction_scores_one() -> None:
    g = _line(5)
    r = evaluate(g, g)
    assert (r.edge_tp, r.edge_fp, r.edge_fn) == (4, 0, 0)


def test_evaluate_datasets_micro_averages() -> None:
    # Two datasets: one perfect, one empty prediction.
    gt = _line(5)
    empty = TrackingGraph()
    res = evaluate_datasets([(gt, gt), (empty, gt)])
    # Micro edge jaccard: tp=4, fp=0, fn=(0)+(4)=4 → 4/8 = 0.5.
    assert res.edge_jaccard == 0.5
    # No divisions anywhere → division term dropped, score == adj edge jaccard.
    assert math.isnan(res.division_jaccard)
    assert res.score == res.adj_edge_jaccard


def test_division_term_enters_score_when_present() -> None:
    # A GT division recovered perfectly by the prediction.
    nodes = {
        0: (0, 0, 0.0, 0), 1: (1, 0, 0.0, 0),
        2: (2, 0, 5.0, 0), 3: (2, 0, -5.0, 0),
        4: (3, 0, 5.0, 0), 5: (3, 0, -5.0, 0),
    }
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5)]
    g = TrackingGraph.from_lists(nodes, edges)
    res = evaluate_datasets([(g, g)], n_true=[6])
    assert res.division_jaccard == 1.0
    assert res.score == res.adj_edge_jaccard + SCORE_DIVISION_WEIGHT * 1.0
