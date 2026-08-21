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


def test_sibling_ratio_gate_suppresses_unbalanced_division():
    # parent at x=0; primary daughter at x=0.5 (d1=0.5), leftover at x=3.0 (d2=3.0)
    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 3.0]]),
    }
    parent0 = 0
    # Gate OFF (ratio=0): the unbalanced leftover still divides (out-degree 2).
    g_off = link_centroids(
        dets, scale=ISO,
        params=LinkParams(max_distance=5.0, allow_division=True,
                          division_distance=5.0, division_max_sibling_ratio=0.0),
    )
    assert g_off.out_degree(parent0) == 2
    # Gate ON (ratio=2.0): d2=3.0 > 2.0*d1=1.0 → sibling rejected, no fork.
    g_on = link_centroids(
        dets, scale=ISO,
        params=LinkParams(max_distance=5.0, allow_division=True,
                          division_distance=5.0, division_max_sibling_ratio=2.0),
    )
    assert g_on.out_degree(parent0) == 1


def test_sibling_ratio_gate_keeps_balanced_division():
    # symmetric split: two daughters at +/-0.5 (d1=d2=0.5) — a real division shape
    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]]),
    }
    g = link_centroids(
        dets, scale=ISO,
        params=LinkParams(max_distance=2.0, allow_division=True,
                          division_distance=2.0, division_max_sibling_ratio=2.0),
    )
    assert g.out_degree(0) == 2  # balanced sibling passes the gate


def test_only_links_consecutive_timepoints():
    dets = {0: np.array([[0.0, 0.0, 0.0]]), 2: np.array([[0.0, 0.0, 0.0]])}
    g = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=5.0))
    assert g.num_edges == 0  # t=0 and t=2 are not consecutive


# --- SOT-2830: global short-window min-cost-flow linking with birth/death arcs ---

def _sorted_edges(g):
    return sorted(g.edges)


def test_global_window_default_is_per_frame_byte_invariant():
    # global_window defaults to 1 => the per-frame champion path is unchanged.
    dets = {
        0: np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]),
        1: np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 11.0]]),
        2: np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 12.0]]),
    }
    base = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    dflt = LinkParams(max_distance=3.0)
    assert dflt.global_window == 1
    assert dflt.birth_cost == float("inf") and dflt.death_cost == float("inf")
    # explicit global_window=1 must be identical to the default path
    g1 = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0, global_window=1))
    assert _sorted_edges(g1) == _sorted_edges(base)


def test_global_window_infinite_theta_reproduces_champion_edges():
    # global path with birth/death = inf (theta = inf) must reproduce the per-frame
    # champion matching exactly, edge-for-edge and in the same insertion order.
    dets = {
        0: np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]),
        1: np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 11.0]]),
        2: np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 12.0]]),
    }
    champ = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    glob = link_centroids(
        dets, scale=ISO,
        params=LinkParams(max_distance=3.0, global_window=2),  # birth=death=inf
    )
    assert glob.edges == champ.edges  # exact list equality (order preserved)


def test_global_birth_death_threshold_suppresses_marginal_link():
    # one source, one dest at scaled distance 3 (<= max_distance=5).
    dets = {0: np.array([[0.0, 0.0, 0.0]]), 1: np.array([[0.0, 0.0, 3.0]])}
    # theta = birth+death = 4.0 > 3 => link kept
    g_keep = link_centroids(
        dets, scale=ISO,
        params=LinkParams(max_distance=5.0, global_window=2, birth_cost=2.0, death_cost=2.0),
    )
    assert g_keep.num_edges == 1
    # theta = 2.0 < 3 => marginal link refused (source dies / dest is born)
    g_drop = link_centroids(
        dets, scale=ISO,
        params=LinkParams(max_distance=5.0, global_window=2, birth_cost=1.0, death_cost=1.0),
    )
    assert g_drop.num_edges == 0
    assert g_drop.num_nodes == 2  # both detections still kept as isolated nodes


def test_global_threshold_frees_dest_for_nearer_source():
    # far source at x=0 (d=4) and near source at x=3 (d=1) both want dest x=4.
    dets = {
        0: np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]),
        1: np.array([[0.0, 0.0, 4.0]]),
    }
    # theta=3 (birth=death=1.5): near link d=1 (<3) kept, far source d=4 unlinked.
    g = link_centroids(
        dets, scale=ISO,
        params=LinkParams(max_distance=6.0, global_window=2, birth_cost=1.5, death_cost=1.5),
    )
    assert g.num_edges == 1
    src = g.edges[0][0]
    assert abs(g.position(src)[2] - 3.0) < 1e-9  # the nearer source won


def test_global_window_output_invariant_to_window_size():
    # for the pure-distance cost model the block flow decouples per transition, so
    # window=2 and window=3 must produce identical graphs (no cross-hop coupling,
    # and no bridge edges leaking across the wider window).
    dets = {
        0: np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]),
        1: np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 11.0]]),
        2: np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 12.0]]),
        3: np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 13.0]]),
    }
    p2 = LinkParams(max_distance=4.0, global_window=2, birth_cost=2.0, death_cost=2.0)
    p3 = LinkParams(max_distance=4.0, global_window=3, birth_cost=2.0, death_cost=2.0)
    g2 = link_centroids(dets, scale=ISO, params=p2)
    g3 = link_centroids(dets, scale=ISO, params=p3)
    assert _sorted_edges(g2) == _sorted_edges(g3)


def test_global_path_emits_only_consecutive_edges():
    # a gap at t=1: t=0 and t=2 must never be bridged on the global path.
    dets = {0: np.array([[0.0, 0.0, 0.0]]), 2: np.array([[0.0, 0.0, 0.0]])}
    g = link_centroids(
        dets, scale=ISO,
        params=LinkParams(max_distance=5.0, global_window=3, birth_cost=10.0, death_cost=10.0),
    )
    assert g.num_edges == 0


# --- SOT-2920: per-track constant-velocity Kalman Mahalanobis LAP gate ---

def test_kalman_gate_default_off_is_byte_invariant():
    # kalman_gate defaults False => the champion path is reproduced edge-for-edge.
    dets = {
        0: np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]),
        1: np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 11.0]]),
        2: np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 12.0]]),
    }
    assert LinkParams().kalman_gate is False
    assert LinkParams().kalman_gate_chi2 == float("inf")
    base = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    off = link_centroids(
        dets, scale=ISO, params=LinkParams(max_distance=3.0, kalman_gate=False)
    )
    assert _sorted_edges(off) == _sorted_edges(base)


def test_kalman_gate_tracks_constant_velocity_cells():
    # three constant-velocity cells; the Kalman path links each into a full chain.
    base = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 30.0], [10.0, 30.0, 10.0]])
    vel = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 1.0, 0.0]])
    dets = {t: base + vel * t for t in range(6)}
    g = link_centroids(
        dets, scale=ISO,
        params=LinkParams(max_distance=7.0, kalman_gate=True, min_track_length=1),
    )
    assert g.num_edges == 3 * 5  # 3 tracks, 5 transitions each
    assert all(g.out_degree(n) <= 1 for n in g.node_ids())


def test_kalman_gate_is_deterministic():
    base = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 30.0]])
    vel = np.array([[0.0, 0.0, 2.0], [0.0, 1.0, 0.0]])
    dets = {t: base + vel * t for t in range(5)}
    p = LinkParams(max_distance=7.0, kalman_gate=True, min_track_length=1)
    a = link_centroids(dets, scale=ISO, params=p)
    b = link_centroids(dets, scale=ISO, params=p)
    assert _sorted_edges(a) == _sorted_edges(b)


def test_kalman_gate_respects_euclidean_cap():
    # a jump beyond max_distance is never admitted (the raw 7 µm cap is preserved
    # by construction, intersected with the Mahalanobis gate).
    dets = {0: np.array([[0.0, 0.0, 0.0]]), 1: np.array([[0.0, 0.0, 100.0]])}
    g = link_centroids(
        dets, scale=ISO, params=LinkParams(max_distance=5.0, kalman_gate=True)
    )
    assert g.num_edges == 0


def test_kalman_gate_chi2_rejects_off_prediction_distractor():
    # a well-established track predicts forward; a finite chi2 gate on a low-noise
    # filter prefers the predicted successor over an equally-near lateral distractor.
    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[0.0, 0.0, 2.0]]),
        2: np.array([[0.0, 0.0, 4.0]]),
        # frame 3: the true forward continuation AND a lateral decoy at equal raw dist
        3: np.array([[0.0, 0.0, 6.0], [0.0, 2.0, 4.0]]),
    }
    g = link_centroids(
        dets, scale=ISO,
        params=LinkParams(
            max_distance=7.0, kalman_gate=True, kalman_process_noise=0.2,
            kalman_gate_chi2=9.0, min_track_length=1,
        ),
    )
    # the track (node 2) must continue to the forward-predicted node, not the decoy
    succ = g.successors(2)
    assert len(succ) == 1
    # node ids: t3 has ids 3 (forward [0,0,6]) and 4 (decoy [0,2,4])
    assert succ[0] == 3
