"""Bidirectional mutual-NN cycle-consistency edge gate (SOT-2910), synthetic.

Data-free: builds detection dicts by hand and checks (1) the gate is byte-for-byte
off by default, (2) a clean isolated mutual-nearest-neighbour link survives, (3) a
globally-assigned but NON-mutual (contested) link the Hungarian optimum makes is
pruned while a well-separated mutual link in the same frame survives, (4) the
ambiguity margin additionally drops a mutual-best link whose runner-up is within
``margin``, and (5) the config round-trips through ``champion_params``. Runs in CI
without the (gitignored) competition volumes.
"""

from __future__ import annotations

import numpy as np

from biohub_tracking.champion import champion_params
from biohub_tracking.link import (
    LinkParams,
    _cycle_consistency_filter,
    link_centroids,
)

ISO = (1.0, 1.0, 1.0)


def test_gate_off_is_byte_for_byte_champion():
    # Two cells drifting +1 in x each frame — every link is unambiguously mutual.
    base = np.array([[0.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    dets = {t: base + np.array([0.0, 0.0, float(t)]) for t in range(4)}
    champ = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    off = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(max_distance=3.0, cycle_consistency_gate=False),
    )
    assert champ.edges == off.edges
    assert champ.num_edges == off.num_edges


def test_clean_mutual_links_survive_when_gate_on():
    # Two well-separated cells drifting +1 in x: each link is a mutual NN, so the
    # gate keeps the champion graph unchanged even when ON.
    base = np.array([[0.0, 0.0, 0.0], [0.0, 40.0, 0.0]])
    dets = {t: base + np.array([0.0, 0.0, float(t)]) for t in range(4)}
    champ = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    gated = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(max_distance=3.0, cycle_consistency_gate=True),
    )
    assert gated.edges == champ.edges
    assert gated.num_edges == champ.num_edges


def test_non_mutual_contested_link_is_pruned():
    # Frame 0 -> 1. A cluster {A=0, B=3} competes for {X=2, Y=5}; a far isolated
    # pair {C=100} -> {Z=100} is unambiguously mutual.
    #   dist: A-X=2 A-Y=5 ; B-X=1 B-Y=2 ; C-Z=0
    # Hungarian optimum (min total) = A->X(2) + B->Y(2) + C->Z(0), but the ONLY
    # mutual pair in the cluster is B<->X, so A->X and B->Y are both non-mutual
    # steals the gate drops; C<->Z survives.
    dets = {
        0: np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 100.0]]),
        1: np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 5.0], [0.0, 0.0, 100.0]]),
    }
    champ = link_centroids(
        dets, scale=ISO, params=LinkParams(max_distance=6.0, allow_division=False)
    )
    assert champ.num_edges == 3  # A->X, B->Y, C->Z
    gated = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(
            max_distance=6.0, allow_division=False, cycle_consistency_gate=True
        ),
    )
    # Only the isolated mutual pair C(id 2) -> Z(id 5) survives.
    assert gated.num_edges == 1
    assert gated.edges == [(2, 5)]


def test_margin_drops_contested_mutual_best():
    # One source A with two close successors X=0.1, Y=0.3. A<->X is a mutual NN
    # (A's nearest is X; X's only source is A), so it survives at margin 0.0 but is
    # contested (runner-up Y within 0.3-0.1=0.2) and dropped at margin 0.3.
    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.3]]),
    }
    keep = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(
            max_distance=3.0,
            allow_division=False,
            cycle_consistency_gate=True,
            cycle_consistency_margin=0.0,
        ),
    )
    assert keep.num_edges == 1  # mutual best A->X survives at margin 0
    drop = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(
            max_distance=3.0,
            allow_division=False,
            cycle_consistency_gate=True,
            cycle_consistency_margin=0.3,
        ),
    )
    assert drop.num_edges == 0  # contested within margin -> pruned


def test_filter_helper_prunes_asymmetric_pairs():
    # Direct helper check on the A/B/X/Y non-mutual configuration.
    src = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    dst = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 5.0]])
    pairs = [(0, 0), (1, 1)]  # Hungarian A->X, B->Y (non-mutual)
    kept = _cycle_consistency_filter(pairs, src, dst, np.array(ISO), margin=0.0)
    assert kept == []  # neither is a mutual NN (B<->X is the only mutual pair)
    # An empty assignment and single-sided frames are returned unchanged.
    assert _cycle_consistency_filter([], src, dst, np.array(ISO), 0.0) == []
    assert (
        _cycle_consistency_filter(
            [(0, 0)], src, np.empty((0, 3)), np.array(ISO), 0.0
        )
        == [(0, 0)]
    )


def test_config_round_trips_through_champion_params():
    cfg = {
        "scale": list(ISO),
        "detect": {},
        "link": {
            "max_distance": 7.0,
            "cycle_consistency_gate": True,
            "cycle_consistency_margin": 1.5,
        },
    }
    _detect, link, _scale = champion_params(cfg)
    assert link.cycle_consistency_gate is True
    assert link.cycle_consistency_margin == 1.5
    # Absent keys keep the champion default (off).
    _d2, link2, _s2 = champion_params({"scale": list(ISO), "detect": {}, "link": {}})
    assert link2.cycle_consistency_gate is False
    assert link2.cycle_consistency_margin == 0.0
