"""Whole-sequence Viterbi global track-linking with swaps (SOT-2918).

Data-free synthetic checks that (1) the mechanism is byte-for-byte off by default and
``viterbi_link=False`` reproduces the champion per-frame graph, (2) with motion off
(gain=0 / curvature_weight=0) and theta=inf the whole-sequence LAP reduces to the same
per-transition distance optimum the champion computes, (3) the birth/death threshold
``viterbi_theta`` refuses a marginal link, (4) the ``<= max_distance`` gate is never
widened by the motion term, (5) only consecutive one-to-one ``t -> t+1`` edges are
emitted, and (6) the champion config builder round-trips the knobs while an absent key
keeps the champion byte-identical. Runs in CI without the (gitignored) competition
volumes.
"""

from __future__ import annotations

import numpy as np

from biohub_tracking.link import LinkParams, link_centroids

ISO = (1.0, 1.0, 1.0)


def _uniform_drift():
    # Three well-separated cells drifting by a constant +1 in x each frame.
    base = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    return {t: base + np.array([0.0, 0.0, float(t)]) for t in range(4)}


def test_viterbi_off_is_byte_for_byte_champion():
    dets = _uniform_drift()
    base = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    dflt = LinkParams(max_distance=3.0)
    assert dflt.viterbi_link is False
    assert dflt.viterbi_theta == float("inf")
    off = link_centroids(dets, scale=ISO, params=dflt)
    assert base.edges == off.edges


def test_viterbi_no_motion_matches_champion_optimum():
    # gain=0 and curvature_weight=0, theta=inf => whole-sequence distance LAP, which on
    # this well-separated drift equals the champion per-frame assignment.
    dets = _uniform_drift()
    base = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    vit = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(
            max_distance=3.0,
            viterbi_link=True,
            viterbi_motion_gain=0.0,
            viterbi_curvature_weight=0.0,
        ),
    )
    assert sorted(base.edges) == sorted(vit.edges)


def test_viterbi_emits_only_consecutive_one_to_one_edges():
    dets = _uniform_drift()
    g = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(max_distance=3.0, viterbi_link=True),
    )
    for a, b in g.edges:
        assert g.coords[b][0] == g.coords[a][0] + 1  # consecutive frames only
    # one-to-one: no source or destination reused
    srcs = [a for a, _ in g.edges]
    dsts = [b for _, b in g.edges]
    assert len(srcs) == len(set(srcs))
    assert len(dsts) == len(set(dsts))


def test_viterbi_theta_refuses_marginal_link():
    # Two frames, a single cell that jumps a distance of 2 (within max_distance=3).
    dets = {0: np.array([[0.0, 0.0, 0.0]]), 1: np.array([[0.0, 0.0, 2.0]])}
    linked = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(max_distance=3.0, viterbi_link=True, viterbi_motion_gain=0.0,
                          viterbi_curvature_weight=0.0),
    )
    assert len(linked.edges) == 1  # theta=inf accepts the feasible link
    # a theta below the effective cost (=2.0 with no motion term) refuses it => birth/death
    refused = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(max_distance=3.0, viterbi_link=True, viterbi_motion_gain=0.0,
                          viterbi_curvature_weight=0.0, viterbi_theta=1.0),
    )
    assert refused.edges == []


def test_viterbi_gate_never_widens_max_distance():
    # A cell whose only candidate is beyond max_distance must never be linked, no matter
    # how strong the motion prediction points at it.
    dets = {0: np.array([[0.0, 0.0, 0.0]]), 1: np.array([[0.0, 0.0, 10.0]])}
    g = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(max_distance=3.0, viterbi_link=True, viterbi_motion_gain=5.0,
                          viterbi_curvature_weight=2.0),
    )
    assert g.edges == []


def test_champion_config_roundtrips_viterbi_knobs():
    from biohub_tracking.champion import champion_params

    cfg = {
        "detect": {},
        "link": {
            "viterbi_link": True,
            "viterbi_motion_gain": 2.0,
            "viterbi_curvature_weight": 0.5,
            "viterbi_theta": 4.0,
            "viterbi_max_sweeps": 3,
        },
        "scale": [1.0, 1.0, 1.0],
    }
    _detect, link, _scale = champion_params(cfg)
    assert link.viterbi_link is True
    assert link.viterbi_motion_gain == 2.0
    assert link.viterbi_curvature_weight == 0.5
    assert link.viterbi_theta == 4.0
    assert link.viterbi_max_sweeps == 3
    # absent key keeps the champion byte-identical (default off, theta unbounded)
    _d2, link2, _s2 = champion_params({"detect": {}, "link": {}, "scale": [1, 1, 1]})
    assert link2.viterbi_link is False
    assert link2.viterbi_theta == float("inf")
