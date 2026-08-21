"""Post-hoc suspicious-tracking-event review gate (SOT-2895), on synthetic tracks.

Data-free: builds detection dicts / graphs by hand and checks (1) the gate is
byte-for-byte off by default, (2) it cuts an interior link that is BOTH a sharp
direction reversal AND a step jump (the teleport-to-decoy signature), and (3) it
leaves a fast-but-straight link and a gentle wobble untouched (the conservative
AND). Runs in CI without the (gitignored) competition volumes.
"""

from __future__ import annotations

import numpy as np

from biohub_tracking.graph import TrackingGraph
from biohub_tracking.link import (
    LinkParams,
    _suspicious_edge_review,
    link_centroids,
)

ISO = (1.0, 1.0, 1.0)


def _chain(xs):
    """A single chain: node i at (t=i, z=0, y=0, x=xs[i]), edges i -> i+1."""
    nodes = {i: (float(i), 0.0, 0.0, float(x)) for i, x in enumerate(xs)}
    edges = [(i, i + 1) for i in range(len(xs) - 1)]
    return TrackingGraph.from_lists(nodes, edges)


def test_reversal_plus_jump_edge_is_cut():
    # Steady +1 drift, then a hard backtrack of -9 at the end: reversal (cos ~ -1)
    # AND jump (9 > 3 * max(1, 1)). The last edge (3 -> 4) must be cut.
    g = _chain([0.0, 1.0, 2.0, 3.0, -6.0])
    out = _suspicious_edge_review(
        g, np.asarray(ISO, dtype=float), turn_cos=-0.5, jump_ratio=3.0, jump_floor=1.0
    )
    assert (3, 4) not in out.edges
    assert set(out.edges) == {(0, 1), (1, 2), (2, 3)}
    # nodes are preserved (only the edge is removed; pruning is a separate step)
    assert out.num_nodes == g.num_nodes


def test_straight_fast_link_is_kept():
    # A big forward step (jump) but NO reversal (cos ~ +1): must be preserved.
    g = _chain([0.0, 1.0, 2.0, 3.0, 20.0])
    out = _suspicious_edge_review(
        g, np.asarray(ISO, dtype=float), turn_cos=-0.5, jump_ratio=3.0, jump_floor=1.0
    )
    assert set(out.edges) == set(g.edges)


def test_gentle_reversal_without_jump_is_kept():
    # A direction reversal but a small step (no jump): must be preserved.
    g = _chain([0.0, 3.0, 6.0, 9.0, 7.0])  # last step -2, |d1|=3 -> 2 < 3*3
    out = _suspicious_edge_review(
        g, np.asarray(ISO, dtype=float), turn_cos=-0.5, jump_ratio=3.0, jump_floor=1.0
    )
    assert set(out.edges) == set(g.edges)


def test_division_vertex_is_never_reviewed():
    # A dividing node (out-degree 2) is excluded from review even if a daughter
    # both reverses and jumps, because the interior-chain guard requires a unique
    # successor.
    nodes = {
        0: (0.0, 0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0, 1.0),
        2: (2.0, 0.0, 0.0, 2.0),
        3: (2.0, 0.0, 0.0, -6.0),  # second daughter: reversal + jump
    }
    edges = [(0, 1), (1, 2), (1, 3)]  # node 1 divides
    g = TrackingGraph.from_lists(nodes, edges)
    out = _suspicious_edge_review(
        g, np.asarray(ISO, dtype=float), turn_cos=-0.5, jump_ratio=3.0, jump_floor=1.0
    )
    assert set(out.edges) == set(g.edges)


def test_gate_off_is_byte_for_byte_champion():
    # A detection stream whose champion linking contains a reversal+jump edge:
    # with the gate OFF the graph is identical; with it ON the decoy edge is gone.
    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[0.0, 0.0, 1.0]]),
        2: np.array([[0.0, 0.0, 2.0]]),
        3: np.array([[0.0, 0.0, 3.0]]),
        4: np.array([[0.0, 0.0, -6.0]]),
    }
    base = LinkParams(max_distance=10.0, allow_division=False, min_track_length=1)
    off = link_centroids(dets, scale=ISO, params=base)

    on = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(
            max_distance=10.0,
            allow_division=False,
            min_track_length=1,
            suspicious_review=True,
        ),
    )
    # champion (off) linked the decoy; the gate (on) removed exactly that edge
    assert (3, 4) in set(off.edges)
    assert (3, 4) not in set(on.edges)
    assert set(off.edges) - set(on.edges) == {(3, 4)}


def test_no_op_when_no_suspicious_edge_returns_same_edges():
    g = _chain([0.0, 1.0, 2.0, 3.0, 4.0])
    out = _suspicious_edge_review(
        g, np.asarray(ISO, dtype=float), turn_cos=-0.5, jump_ratio=3.0, jump_floor=1.0
    )
    assert set(out.edges) == set(g.edges)
    assert out.num_nodes == g.num_nodes
