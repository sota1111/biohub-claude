"""Tests for semi-supervised dense pseudo-label self-training (SOT-2996).

Data-free: synthetic candidates + a tiny GT graph, so the labeling logic and the
champion-track consistency gate are pinned without the (gitignored) competition data.
"""

from __future__ import annotations

import numpy as np

from biohub_tracking.detect_scorer import FEATURE_NAMES
from biohub_tracking.graph import TrackingGraph
from biohub_tracking.link import LinkParams
from biohub_tracking.pseudo_label import (
    PseudoLabelConfig,
    densify_labels,
    track_length_per_candidate,
)

SCALE = (1.0, 1.0, 1.0)
RESP_COL = FEATURE_NAMES.index("resp_z")


def _feats(resp_z_values: list[float]) -> np.ndarray:
    """Feature matrix whose resp_z column is set; other columns are zeros."""
    x = np.zeros((len(resp_z_values), len(FEATURE_NAMES)), dtype=np.float64)
    x[:, RESP_COL] = resp_z_values
    return x


def _gt_at(points_by_t: dict[int, list[tuple[float, float, float]]]) -> TrackingGraph:
    """A GT graph with nodes only (positions matter, edges irrelevant here)."""
    g = TrackingGraph()
    nid = 0
    for t, pts in points_by_t.items():
        for (z, y, x) in pts:
            g.add_node(nid, float(t), z, y, x)
            nid += 1
    return g


def test_track_length_flags_persistent_blob():
    """A blob present and moving slightly across 5 frames forms one component of 5;
    an isolated singleton stays length 1."""
    dets = {}
    for t in range(5):
        # persistent blob drifting by 1 voxel/frame + a far-away singleton each frame
        # only at t=0 so it cannot link forward.
        rows = [(10.0 + t, 10.0, 10.0)]
        if t == 0:
            rows.append((0.0, 30.0, 30.0))
        dets[t] = np.array(rows, dtype=np.float64)
    tl = track_length_per_candidate(dets, SCALE, LinkParams())
    # row order = sorted-t, det-order. t=0 has [blob, singleton]; t1..4 one each.
    assert tl[0] == 5  # persistent blob component
    assert tl[1] == 1  # the isolated singleton at t=0


def test_track_length_one_row_per_candidate_when_params_prune():
    """Regression (SOT-2996): when the passed champion params would prune short
    tracks (``min_track_length > 1``) or add interpolated nodes, the returned track
    length must still have exactly one row per candidate in row_t order. The linker's
    prune drops nodes / re-numbers ids, so relying on the pruned graph's node count
    produced ``track_len rows != candidate rows`` and crashed the screen."""
    dets = {}
    n_expected = 0
    for t in range(5):
        rows = [(10.0 + t, 10.0, 10.0)]  # persistent blob (survives a prune)
        if t == 0:
            rows.append((0.0, 30.0, 30.0))  # singleton (a prune would DROP this)
        arr = np.array(rows, dtype=np.float64)
        dets[t] = arr
        n_expected += arr.shape[0]
    # Champion-like params that prune singletons and would interpolate gap nodes.
    pruning_params = LinkParams(min_track_length=3, gap_recover=True)
    tl = track_length_per_candidate(dets, SCALE, pruning_params)
    assert tl.shape[0] == n_expected  # one row per candidate, nothing dropped/added
    assert tl[0] == 5  # persistent blob component still measured
    assert tl[1] == 1  # the singleton is kept as a length-1 track, not pruned away


def test_densify_promotes_consistent_highconf_unmatched():
    """An unmatched, high-confidence, temporally-consistent candidate becomes a
    pseudo-positive; a GT-matched one stays positive; a lonely high-conf speck and a
    low-conf candidate stay negative."""
    # Persistent real cell (unannotated) at (10+t,10,10) across 5 frames, high resp.
    # GT annotates a *different* lineage at (10+t,40,40).
    dets = {}
    resp = []
    for t in range(5):
        rows = [(10.0 + t, 10.0, 10.0)]  # persistent unmatched cell -> pseudo +
        vals = [5.0]
        if t == 0:
            rows.append((0.0, 25.0, 25.0))  # lonely high-conf speck -> negative
            vals.append(5.0)
            rows.append((0.0, 5.0, 5.0))  # low-conf -> negative
            vals.append(-2.0)
            rows.append((10.0, 40.0, 40.0))  # GT-matched -> positive
            vals.append(5.0)
        dets[t] = np.array(rows, dtype=np.float64)
        resp.extend(vals)
    feats = _feats(resp)
    gt = _gt_at({0: [(10.0, 40.0, 40.0)]})

    cfg = PseudoLabelConfig(resp_percentile=50.0, min_track_len=3, pseudo_weight=0.4)
    labels, weights, diag = densify_labels(
        dets, feats, FEATURE_NAMES, gt, SCALE, LinkParams(), cfg
    )
    # row order: t0 = [persistent, speck, lowconf, gtmatch], then t1..4 persistent.
    assert labels[0] == 1.0 and weights[0] == cfg.pseudo_weight  # pseudo-positive
    assert labels[3] == 1.0 and weights[3] == 1.0  # GT match, full weight
    assert labels[1] == 0.0  # lonely speck fails consistency -> negative
    assert labels[2] == 0.0  # low confidence -> negative
    assert diag.n_gt_positive == 1
    assert diag.n_pseudo_positive >= 1
    assert diag.densification_ratio > 1.0


def test_feature_name_guard():
    dets = {0: np.array([[1.0, 1.0, 1.0]])}
    feats = np.zeros((1, 3))
    gt = _gt_at({0: [(1.0, 1.0, 1.0)]})
    try:
        densify_labels(dets, feats, ("a", "b", "c"), gt, SCALE, LinkParams())
    except ValueError as e:
        assert "resp_z" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing resp_z feature")


def test_empty_family_is_safe():
    labels, weights, diag = densify_labels(
        {}, np.zeros((0, len(FEATURE_NAMES))), FEATURE_NAMES, _gt_at({}), SCALE,
        LinkParams(),
    )
    assert labels.shape[0] == 0 and weights.shape[0] == 0
    assert diag.n_candidates == 0
