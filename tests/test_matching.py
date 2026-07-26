"""Node-matching behaviour: per-timepoint, one-to-one, scaled, thresholded."""

from __future__ import annotations

from biohub_tracking.graph import TrackingGraph
from biohub_tracking.matching import match_nodes


def _g(nodes, edges=()):
    return TrackingGraph.from_lists(nodes, list(edges))


def test_matches_only_within_same_timepoint() -> None:
    # A pred node close in space but at a different timepoint must not match.
    gt = _g({1: (0, 0, 0, 0)})
    pred = _g({10: (1, 0, 0, 0)})  # same position, different t
    assert match_nodes(pred, gt, scale=None, max_distance=7.0) == {}


def test_matches_nearest_within_frame() -> None:
    gt = _g({1: (0, 0, 0.0, 0), 2: (0, 0, 20.0, 0)})
    pred = _g({10: (0, 0, 1.0, 0), 11: (0, 0, 21.0, 0)})
    assert match_nodes(pred, gt, scale=None, max_distance=7.0) == {10: 1, 11: 2}


def test_threshold_excludes_far_pairs() -> None:
    gt = _g({1: (0, 0, 0.0, 0)})
    pred = _g({10: (0, 0, 8.0, 0)})  # 8 > 7 µm
    assert match_nodes(pred, gt, scale=None, max_distance=7.0) == {}
    assert match_nodes(pred, gt, scale=None, max_distance=10.0) == {10: 1}


def test_scale_is_applied_to_distance() -> None:
    # dz = 5 voxels; with z-scale 1.625 the physical distance is 8.125 µm > 7.
    gt = _g({1: (0, 0.0, 0, 0)})
    pred = _g({10: (0, 5.0, 0, 0)})
    assert match_nodes(pred, gt, scale=(1.625, 0.40625, 0.40625), max_distance=7.0) == {}
    # Unscaled (isotropic) the same offset is only 5 µm < 7 → matches.
    assert match_nodes(pred, gt, scale=None, max_distance=7.0) == {10: 1}


def test_assignment_is_one_to_one() -> None:
    gt = _g({1: (0, 0, 0.0, 0), 2: (0, 0, 1.0, 0)})
    pred = _g({10: (0, 0, 0.4, 0)})  # near both, but only one match allowed
    m = match_nodes(pred, gt, scale=None, max_distance=7.0)
    assert m == {10: 1}


def test_tie_prefers_in_threshold_pair() -> None:
    # Total cost ties between {far,far} and {near,far}; masking must keep the
    # near, in-threshold pair rather than dropping both.
    gt = _g({1: (1, 0, 10.0, 0), 2: (1, 0, -40.0, 0)})
    pred = _g({10: (1, 0, 55.0, 0), 11: (1, 0, 15.0, 0)})
    m = match_nodes(pred, gt, scale=None, max_distance=10.0)
    assert m == {11: 1}  # 11↔1 is dist 5; every other pair is > 10


def test_gt_node_subset_restricts_targets() -> None:
    gt = _g({1: (0, 0, 0.0, 0), 2: (0, 0, 1.0, 0)})
    pred = _g({10: (0, 0, 0.9, 0)})
    # Without restriction, 10 matches the nearer GT node 2.
    assert match_nodes(pred, gt, scale=None, max_distance=7.0) == {10: 2}
    # Restricted to {1}, it must match 1 instead.
    assert match_nodes(pred, gt, scale=None, max_distance=7.0, gt_node_subset={1}) == {10: 1}
