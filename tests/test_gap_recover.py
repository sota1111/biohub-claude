"""Node-interpolation gap recovery (SOT-2849), on synthetic centroid tracks.

Data-free: builds detection dicts by hand and checks the interpolated-node
topology, its distinction from the SOT-2763 skip-bridge (consecutive edges only),
the fragment-size gate, and byte-for-byte off-by-default, so it runs in CI without
the (gitignored) competition volumes.
"""

from __future__ import annotations

import numpy as np

from biohub_tracking.link import LinkParams, link_centroids

# isotropic scale keeps the arithmetic transparent in these tests
ISO = (1.0, 1.0, 1.0)


def _three_plus_three_with_gap():
    # A cell present in frames 0,1,2, missing at 3, then present 4,5,6 — two real
    # 3-node fragments separated by one missing frame.
    return {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[0.0, 0.0, 1.0]]),
        2: np.array([[0.0, 0.0, 2.0]]),
        4: np.array([[0.0, 0.0, 4.0]]),
        5: np.array([[0.0, 0.0, 5.0]]),
        6: np.array([[0.0, 0.0, 6.0]]),
    }


def test_gap_recover_off_is_byte_for_byte_champion():
    dets = _three_plus_three_with_gap()
    base = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=2.0))
    dflt = link_centroids(
        dets, scale=ISO, params=LinkParams(max_distance=2.0, gap_recover=False)
    )
    # Two fragments (0-1-2 and 4-5-6): 4 internal consecutive edges, no bridge/node.
    assert base.num_edges == 4
    assert base.num_nodes == 6
    assert dflt.num_edges == base.num_edges and dflt.num_nodes == base.num_nodes


def test_gap_recover_inserts_interpolated_node_and_consecutive_edges():
    dets = _three_plus_three_with_gap()
    g = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(
            max_distance=2.0,
            gap_recover=True,
            gap_recover_max_gap=2,
            gap_recover_distance=7.0,
            gap_recover_min_frag=1,
        ),
    )
    # One interpolated node added at the missing frame 3, wiring 2->interp->4.
    assert g.num_nodes == 7
    # 4 original internal edges + 2 recovery edges (tail->interp, interp->head).
    assert g.num_edges == 6
    # The recovered path is fully consecutive (metric-scored), unlike a skip bridge.
    times = sorted(int(t) for t in [g.coords[n][0] for n in g.node_ids()])
    assert times == [0, 1, 2, 3, 4, 5, 6]
    # The interpolated node sits at frame 3, linearly between (0,0,2) and (0,0,4).
    interp = [n for n in g.node_ids() if int(g.coords[n][0]) == 3]
    assert len(interp) == 1
    assert np.allclose(g.position(interp[0]), [0.0, 0.0, 3.0])
    # It is wired into one weakly-connected 7-node track (in==out==1 in the middle).
    assert g.in_degree(interp[0]) == 1 and g.out_degree(interp[0]) == 1


def test_gap_recover_min_frag_gate_refuses_short_fragments():
    # Two singleton fragments separated by a gap: with min_frag=2 neither terminal
    # is eligible, so nothing is bridged (the anti-noise-resurrection gate).
    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        2: np.array([[0.0, 0.0, 2.0]]),
    }
    g = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(
            max_distance=2.0,
            gap_recover=True,
            gap_recover_max_gap=2,
            gap_recover_distance=7.0,
            gap_recover_min_frag=2,
        ),
    )
    assert g.num_nodes == 2 and g.num_edges == 0


def test_gap_recover_respects_distance_gate():
    # Tail->head scaled distance exceeds gap_recover_distance -> no bridge.
    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[0.0, 0.0, 1.0]]),
        3: np.array([[0.0, 0.0, 20.0]]),
        4: np.array([[0.0, 0.0, 21.0]]),
    }
    g = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(
            max_distance=2.0,
            gap_recover=True,
            gap_recover_max_gap=2,
            gap_recover_distance=7.0,
            gap_recover_min_frag=1,
        ),
    )
    # No interpolated node inserted (20 um >> 7 um gate); two 2-node fragments.
    assert g.num_nodes == 4 and g.num_edges == 2
