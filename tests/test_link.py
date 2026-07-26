"""Frame-to-frame linking on synthetic centroid tracks."""

from __future__ import annotations

import numpy as np

from biohub_tracking.link import LinkParams, link_centroids

# isotropic scale keeps the arithmetic transparent in these tests
ISO = (1.0, 1.0, 1.0)


def test_links_moving_cell_across_frames():
    # one cell drifting 1 voxel/frame in x
    dets = {t: np.array([[5.0, 5.0, 5.0 + t]]) for t in range(4)}
    g = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=2.0))
    assert g.num_nodes == 4
    assert g.num_edges == 3  # a chain
    # each node has out-degree <= 1 (no spurious divisions)
    assert all(g.out_degree(n) <= 1 for n in g.node_ids())


def test_breaks_track_beyond_max_distance():
    dets = {0: np.array([[0.0, 0.0, 0.0]]), 1: np.array([[0.0, 0.0, 100.0]])}
    g = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=5.0))
    assert g.num_edges == 0  # too far → no link


def test_optimal_assignment_pairs_nearest():
    # two cells that must not cross-link
    dets = {
        0: np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]),
        1: np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 11.0]]),
    }
    g = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    assert g.num_edges == 2
    ids0 = g.nodes_by_time()[0]
    # node at x=0 links to x=1, node at x=10 links to x=11
    for s in ids0:
        succ = g.successors(s)
        assert len(succ) == 1
        assert abs(g.position(s)[2] - g.position(succ[0])[2]) <= 1.0


def test_division_attaches_second_daughter_when_enabled():
    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]]),
    }
    g_on = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=2.0, allow_division=True))
    parent = g_on.nodes_by_time()[0][0]
    assert g_on.out_degree(parent) == 2  # 1 -> 2 split

    g_off = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=2.0, allow_division=False))
    parent = g_off.nodes_by_time()[0][0]
    assert g_off.out_degree(parent) == 1  # no division when disabled


def test_only_links_consecutive_timepoints():
    dets = {0: np.array([[0.0, 0.0, 0.0]]), 2: np.array([[0.0, 0.0, 0.0]])}
    g = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=5.0))
    assert g.num_edges == 0  # t=0 and t=2 are not consecutive
