"""Motion-coupled windowed global association + parental-softmax (SOT-2871).

Data-free synthetic checks that (1) the mechanism is byte-for-byte off by default
and W=1 equals the champion, (2) with motion off and theta=inf the windowed LAP
chain reduces to the per-transition champion assignment, (3) the birth/death
threshold suppresses a marginal link, (4) the carried window velocity actually
*bites* — it links a fast, motion-consistent successor the memoryless champion
mislinks to a nearer decoy, (5) the windowed path emits only consecutive edges and
resets its motion history across a frame gap, and (6) parental-softmax attaches a
balanced second daughter while rejecting a distant fake sibling. Runs in CI without
the (gitignored) competition volumes.
"""

from __future__ import annotations

import numpy as np

from biohub_tracking.link import LinkParams, link_centroids

# isotropic scale keeps the arithmetic transparent in these tests
ISO = (1.0, 1.0, 1.0)


def _uniform_drift():
    # Three cells drifting by a constant +1 in x each frame (a rigid translation).
    base = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    return {t: base + np.array([0.0, 0.0, float(t)]) for t in range(4)}


def test_window_off_is_byte_for_byte_champion():
    dets = _uniform_drift()
    base = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    dflt = LinkParams(max_distance=3.0)
    assert dflt.window_assoc == 1
    assert dflt.window_theta == float("inf")
    off = link_centroids(dets, scale=ISO, params=dflt)
    assert base.edges == off.edges
    # explicit window_assoc=1 must be identical to the default per-frame path
    w1 = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0, window_assoc=1))
    assert base.edges == w1.edges


def test_window_motion_off_reduces_to_champion_assignment():
    # gain 0 => src_pred is None => the windowed LAP is the champion _assign per
    # transition, and theta=inf accepts every feasible pair, so edges match.
    dets = _uniform_drift()
    champ = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    win = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(max_distance=3.0, window_assoc=2, motion_gain=0.0),
    )
    assert champ.edges == win.edges


def test_window_birth_death_threshold_suppresses_marginal_link():
    # One feasible pair at scaled distance 3 (<= max_distance 5). theta > 3 links it;
    # theta < 3 refuses it (the source dies / the destination is born).
    dets = {0: np.array([[0.0, 0.0, 0.0]]), 1: np.array([[0.0, 0.0, 3.0]])}
    linked = link_centroids(
        dets, scale=ISO,
        params=LinkParams(max_distance=5.0, window_assoc=2, motion_gain=0.0, window_theta=4.0),
    )
    assert linked.num_edges == 1
    refused = link_centroids(
        dets, scale=ISO,
        params=LinkParams(max_distance=5.0, window_assoc=2, motion_gain=0.0, window_theta=2.0),
    )
    assert refused.num_edges == 0


def test_carried_velocity_bites_vs_memoryless_champion():
    # A cell moving +3 in x per frame, then a decoy sits right next to its t1
    # position. The memoryless champion links to the nearer decoy; the windowed
    # carried velocity predicts the fast successor and links it instead.
    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[0.0, 0.0, 3.0]]),
        2: np.array([[0.0, 0.0, 6.0], [0.0, 0.0, 3.2]]),  # node 2 = real, node 3 = decoy
    }
    champ = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=5.0))
    assert (1, 3) in champ.edges  # champion links the nearer decoy
    win = link_centroids(
        dets, scale=ISO,
        params=LinkParams(
            max_distance=5.0, window_assoc=2, motion_gain=1.0, window_carry_weight=1.0
        ),
    )
    assert (1, 2) in win.edges  # window follows the trajectory to the real successor
    assert (1, 3) not in win.edges


def test_window_emits_only_consecutive_edges_and_resets_across_gap():
    # Frames 0,1,3 (a missing frame 2). Only the 0->1 transition is consecutive;
    # 1->3 must not be bridged (only metric-valid consecutive edges are emitted).
    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[0.0, 0.0, 1.0]]),
        3: np.array([[0.0, 0.0, 2.0]]),
    }
    g = link_centroids(
        dets, scale=ISO, params=LinkParams(max_distance=5.0, window_assoc=3, motion_gain=1.0)
    )
    for s, d in g.edges:
        assert g.t(d) - g.t(s) == 1


def test_parental_softmax_attaches_balanced_daughter_not_distant_decoy():
    # Parent at origin; two balanced daughters at +/-2; a distant decoy at 4.5.
    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0], [0.0, 0.0, 4.5]]),
    }
    params = LinkParams(
        max_distance=5.0, window_assoc=2, motion_gain=0.0,
        allow_division=True, division_distance=5.0,
        window_parental_softmax=True, window_softmax_min_share=0.3,
        window_softmax_temp=1.0,
    )
    g = link_centroids(dets, scale=ISO, params=params)
    parent = 0
    assert g.out_degree(parent) == 2  # primary + one balanced second daughter
    # the distant decoy (node 3) is never attached as a fake sibling
    assert g.in_degree(3) == 0
    # without parental-softmax the second daughter is not added
    no_soft = link_centroids(
        dets, scale=ISO,
        params=LinkParams(
            max_distance=5.0, window_assoc=2, motion_gain=0.0,
            allow_division=True, division_distance=5.0,
            window_parental_softmax=False,
        ),
    )
    assert no_soft.out_degree(parent) == 1


def test_windowed_association_is_deterministic():
    dets = _uniform_drift()
    params = LinkParams(
        max_distance=3.0, window_assoc=3, motion_gain=1.0,
        motion_smooth_sigma=15.0, window_carry_weight=0.5,
        motion_gate_on_prediction=True,
    )
    a = link_centroids(dets, scale=ISO, params=params)
    b = link_centroids(dets, scale=ISO, params=params)
    assert a.edges == b.edges
