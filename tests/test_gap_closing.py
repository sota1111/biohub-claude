"""Gap-closing 2nd LAP linking step (SOT-2763), on synthetic centroid tracks.

Data-free: builds detection dicts by hand and checks the bridging topology and
its interaction with short-track pruning, so it runs in CI without the
(gitignored) competition volumes.
"""

from __future__ import annotations

import numpy as np

from biohub_tracking.link import LinkParams, link_centroids

# isotropic scale keeps the arithmetic transparent in these tests
ISO = (1.0, 1.0, 1.0)


def test_gap_close_off_is_byte_for_byte_champion():
    # A cell present in frames 0,1,2 then missing at 3 then 4,5. With gap-closing
    # off (max_frame_gap=1) the track stays split into two fragments (2 edges).
    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[0.0, 0.0, 1.0]]),
        2: np.array([[0.0, 0.0, 2.0]]),
        4: np.array([[0.0, 0.0, 4.0]]),
        5: np.array([[0.0, 0.0, 5.0]]),
    }
    g = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=2.0))
    assert g.num_edges == 3  # 0->1->2 (2) and 4->5 (1), no bridge
    assert g.num_nodes == 5


def test_gap_close_bridges_across_one_missing_frame():
    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[0.0, 0.0, 1.0]]),
        2: np.array([[0.0, 0.0, 2.0]]),
        4: np.array([[0.0, 0.0, 4.0]]),
        5: np.array([[0.0, 0.0, 5.0]]),
    }
    g = link_centroids(
        dets, scale=ISO,
        params=LinkParams(max_distance=2.0, max_frame_gap=2, gap_distance=5.0),
    )
    # 3 consecutive edges (0->1->2, 4->5) + 1 bridge edge (t=2 -> t=4).
    assert g.num_edges == 4
    tail = [n for n in g.node_ids() if g.t(n) == 2][0]
    head = [n for n in g.node_ids() if g.t(n) == 4][0]
    assert head in g.successors(tail)


def test_gap_distance_gate_rejects_far_bridge():
    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[0.0, 0.0, 1.0]]),
        3: np.array([[0.0, 0.0, 100.0]]),  # far away head, gap of 2
        4: np.array([[0.0, 0.0, 101.0]]),
    }
    g = link_centroids(
        dets, scale=ISO,
        params=LinkParams(max_distance=2.0, max_frame_gap=3, gap_distance=5.0),
    )
    assert g.num_edges == 2  # no bridge: 100µm >> gap_distance


def test_frame_gap_window_respected():
    # tail at t=0, head at t=3 -> gap g=3, only bridged if max_frame_gap>=3.
    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        3: np.array([[0.0, 0.0, 1.0]]),
        4: np.array([[0.0, 0.0, 2.0]]),
    }
    g2 = link_centroids(
        dets, scale=ISO,
        params=LinkParams(max_distance=2.0, max_frame_gap=2, gap_distance=5.0),
    )
    assert g2.num_edges == 1  # gap of 3 exceeds max_frame_gap=2 -> no bridge
    g3 = link_centroids(
        dets, scale=ISO,
        params=LinkParams(max_distance=2.0, max_frame_gap=3, gap_distance=5.0),
    )
    assert g3.num_edges == 2  # bridge admitted at g=3


def test_gap_close_rescues_fragment_from_short_track_prune():
    # Two 3-node fragments split by a 1-frame gap. Alone each has <4 nodes so
    # min_track_length=4 prunes BOTH (dropping every real edge). Gap-closing
    # bridges them into a 6-node component that survives, rescuing the internal
    # consecutive edges.
    dets = {t: np.array([[0.0, 0.0, float(t)]]) for t in [0, 1, 2, 4, 5, 6]}
    pruned = link_centroids(
        dets, scale=ISO, params=LinkParams(max_distance=2.0, min_track_length=4),
    )
    assert pruned.num_nodes == 0  # both fragments pruned, all edges lost

    rescued = link_centroids(
        dets, scale=ISO,
        params=LinkParams(
            max_distance=2.0, max_frame_gap=2, gap_distance=5.0, min_track_length=4
        ),
    )
    assert rescued.num_nodes == 6  # component survives the prune
    # 4 consecutive edges (0->1->2, 4->5->6) recovered + 1 bridge (2->4).
    consecutive = [
        (s, t) for s, t in rescued.edges if rescued.t(t) - rescued.t(s) == 1
    ]
    assert len(consecutive) == 4


def test_bridges_are_one_to_one_optimal():
    # Two tails and two heads in range; the optimal assignment must pair each
    # tail with its nearest head, not cross-link, and never fork a terminal.
    dets = {
        0: np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]]),
        1: np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 21.0]]),
        3: np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 22.0]]),
        4: np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 23.0]]),
    }
    g = link_centroids(
        dets, scale=ISO,
        params=LinkParams(max_distance=2.0, max_frame_gap=2, gap_distance=5.0),
    )
    # each tail (t=1) gets exactly one successor; no terminal forks
    for n in g.node_ids():
        assert g.out_degree(n) <= 1
        assert g.in_degree(n) <= 1
    # nearest-pairing: the x~1 tail bridges to the x~2 head (not the x~22 head)
    tail_lo = [n for n in g.node_ids() if g.t(n) == 1 and g.position(n)[2] < 10][0]
    succ = g.successors(tail_lo)
    assert len(succ) == 1
    assert g.position(succ[0])[2] < 10
