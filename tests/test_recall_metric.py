"""Unit tests for the GT-node recall @7 µm evaluator (SOT-2873)."""

from __future__ import annotations

import math

from biohub_tracking.eval.recall_metric import gt_node_recall
from biohub_tracking.graph import TrackingGraph


def _g(nodes, edges=()):
    return TrackingGraph.from_lists(nodes, list(edges))


def test_full_coverage():
    """Every GT node matched → recall 1.0 on both denominators."""
    gt = _g({1: (0, 0, 0, 0), 2: (1, 0, 0, 0)}, edges=[(1, 2)])
    pred = _g({10: (0, 0, 0.5, 0), 11: (1, 0, 0.5, 0)}, edges=[(10, 11)])
    r = gt_node_recall(pred, gt, scale=None, max_distance=7.0)
    assert r.gt_nodes == 2 and r.gt_nodes_matched == 2
    assert r.gt_node_recall == 1.0
    assert r.gt_edge_endpoint_nodes == 2 and r.gt_edge_endpoint_recall == 1.0


def test_partial_endpoint_recall():
    """A missed edge endpoint lowers the endpoint recall, matching the metric view."""
    gt = _g(
        {1: (0, 0, 0, 0), 2: (1, 0, 0, 0), 3: (1, 0, 40.0, 0)},
        edges=[(1, 2)],  # node 3 is isolated (no incident edge)
    )
    # Predict only node 1's position (endpoint), miss node 2 (>7µm), miss node 3.
    pred = _g({10: (0, 0, 0.2, 0)})
    r = gt_node_recall(pred, gt, scale=None, max_distance=7.0)
    assert r.gt_nodes == 3 and r.gt_nodes_matched == 1
    assert math.isclose(r.gt_node_recall, 1 / 3)
    # Endpoint denominator excludes the isolated node 3 → 2 endpoints, 1 covered.
    assert r.gt_edge_endpoint_nodes == 2
    assert r.gt_edge_endpoint_matched == 1
    assert r.gt_edge_endpoint_recall == 0.5


def test_isolated_nodes_excluded_from_endpoint_denominator():
    """GT nodes with no incident GT edge do not count toward endpoint recall."""
    gt = _g({1: (0, 0, 0, 0), 2: (0, 0, 40.0, 0)})  # two isolated nodes, no edges
    pred = _g({10: (0, 0, 0.1, 0)})
    r = gt_node_recall(pred, gt, scale=None, max_distance=7.0)
    assert r.gt_node_recall == 0.5
    assert r.gt_edge_endpoint_nodes == 0
    assert math.isnan(r.gt_edge_endpoint_recall)


def test_empty_pred_zero_recall():
    gt = _g({1: (0, 0, 0, 0), 2: (1, 0, 0, 0)}, edges=[(1, 2)])
    pred = _g({})
    r = gt_node_recall(pred, gt, scale=None, max_distance=7.0)
    assert r.gt_nodes_matched == 0
    assert r.gt_node_recall == 0.0
    assert r.gt_edge_endpoint_recall == 0.0
