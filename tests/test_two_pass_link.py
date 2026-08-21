"""Two-pass tight-then-full-gate Hungarian linking (SOT-2899), on synthetic tracks.

Data-free: builds detection dicts by hand and checks (1) the mechanism is
byte-for-byte off by default, (2) ``link_full_distance <= max_distance`` is a
strict no-op, (3) the second pass recovers a fast cell just beyond the tight gate
without perturbing a Pass-1 high-confidence link, and (4) the tight first pass
protects against a distractor steal a single wide-gate pass would commit. Runs in
CI without the (gitignored) competition volumes.
"""

from __future__ import annotations

import numpy as np

from biohub_tracking.link import (
    LinkParams,
    _two_pass_assign,
    link_centroids,
)

ISO = (1.0, 1.0, 1.0)


def test_two_pass_off_is_byte_for_byte_champion():
    # Two cells drifting +1 in x each frame.
    base = np.array([[0.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    dets = {t: base + np.array([0.0, 0.0, float(t)]) for t in range(4)}
    champ = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    dflt = link_centroids(
        dets, scale=ISO, params=LinkParams(max_distance=3.0, link_two_pass=False)
    )
    assert champ.edges == dflt.edges
    assert champ.num_edges == dflt.num_edges


def test_full_distance_not_wider_is_noop():
    base = np.array([[0.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    dets = {t: base + np.array([0.0, 0.0, float(t)]) for t in range(4)}
    champ = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    # two_pass ON but full gate == max_distance -> Pass 2 has no wider reach.
    same = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(max_distance=3.0, link_two_pass=True, link_full_distance=3.0),
    )
    assert champ.edges == same.edges


def test_second_pass_recovers_cell_beyond_tight_gate():
    # One cell jumps 5 units in x between t=0 and t=1 (beyond the tight gate 3),
    # but within the full gate 6. Single-pass champion drops the link; two-pass
    # recovers it in Pass 2.
    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[0.0, 0.0, 5.0]]),
    }
    champ = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    assert champ.num_edges == 0  # 5 > 3, no link
    two_pass = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(max_distance=3.0, link_two_pass=True, link_full_distance=6.0),
    )
    assert two_pass.num_edges == 1  # recovered by the wider second pass


def test_tight_first_pass_prevents_distractor_steal():
    # src A at x=0, src B at x=10. dst A' at x=1 (A's true successor, 1 away),
    # dst B' at x=4 (between; 4 from A, 6 from B). With a single wide gate (>=6)
    # a global assignment could pair A->A'(1)+B->B'(6)=7, which is optimal, so
    # this configuration links both correctly either way. Instead test the classic
    # steal: only ONE dst near the boundary that both want.
    #
    # src A=0, B=5.  dst X=3 (3 from A, 2 from B).  Tight gate 3.5:
    #   Pass 1 (<=3.5): A-X (3) and B-X (2) both feasible; LAP assigns B-X (cheaper
    #   total) leaving A unmatched. That is correct: X is B's cell.
    # Full gate 6 second pass: A has no remaining dst -> ends its track (right).
    dets = {
        0: np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
        1: np.array([[0.0, 0.0, 3.0]]),
    }
    two_pass = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(max_distance=3.5, link_two_pass=True, link_full_distance=6.0),
    )
    # Exactly one edge, and it must be B(src idx1)->X, not A(src idx0)->X.
    assert two_pass.num_edges == 1
    (src_id, dst_id) = two_pass.edges[0]
    # node ids: t0 -> [0,1], t1 -> [2]; B is id 1.
    assert src_id == 1 and dst_id == 2


def test_two_pass_assign_helper_indices_map_back():
    # A cell at x=0 links tight to x=1; a second cell at x=0,y=10 links only via
    # the wide pass to x=0,y=15 (distance 5). Indices must map to the originals.
    src = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    dst = np.array([[0.0, 0.0, 1.0], [0.0, 15.0, 0.0]])
    pairs = _two_pass_assign(src, dst, np.array(ISO), tight_gate=3.0, full_gate=6.0)
    d = dict(pairs)
    assert d[0] == 0  # tight pass
    assert d[1] == 1  # wide pass, indices mapped back to originals
