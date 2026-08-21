"""Unit tests for the non-destructive division overlay (SOT-2818).

Pure / data-free: the overlay is a deterministic transform of a
:class:`~biohub_tracking.graph.TrackingGraph`, so every case is a hand-built
graph — no competition data required.
"""

from __future__ import annotations

import numpy as np

from biohub_tracking.division_overlay import apply_division_overlay
from biohub_tracking.eval import division_counts
from biohub_tracking.graph import TrackingGraph
from biohub_tracking.link import LinkParams, link_centroids

SCALE = (1.0, 1.0, 1.0)
# ("kind", max_distance, sibling_ratio, min_daughter_len, require_parent_track)
DEFAULT = ("nearest-head", 7.0, 2.0, 2, True)


def _division_scenario() -> TrackingGraph:
    """Parent track P0->P1->P2 splitting into two persistent daughters at t=3.

    The champion (one-to-one linker) attaches P2 to the primary daughter chain
    (y=-1) and leaves the second daughter chain (y=+1) as an unlinked head — the
    exact shape the overlay is meant to re-attach.
    """
    nodes = {
        0: (0, 0, 0, 0),   # P0
        1: (1, 0, 0, 0),   # P1
        2: (2, 0, 0, 0),   # P2 (parent: has predecessor, one successor)
        3: (3, 0, -1, 0),  # primary daughter head (linked)
        4: (4, 0, -1, 0),
        5: (5, 0, -1, 0),
        6: (3, 0, 1, 0),   # second daughter head (dropped by the linker)
        7: (4, 0, 1, 0),
        8: (5, 0, 1, 0),
    }
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8)]
    return TrackingGraph.from_lists(nodes, edges)


def test_off_is_a_no_op_byte_for_byte() -> None:
    g = _division_scenario()
    before_nodes = dict(g.coords)
    before_edges = list(g.edges)
    out = apply_division_overlay(g, SCALE, None)
    assert out is g
    assert out.coords == before_nodes
    assert out.edges == before_edges
    # An empty tuple disables it too.
    apply_division_overlay(g, SCALE, ())
    assert g.edges == before_edges


def test_overlay_reattaches_dropped_daughter_and_makes_a_fork() -> None:
    g = _division_scenario()
    apply_division_overlay(g, SCALE, DEFAULT)
    # Exactly one edge added: the parent's dropped second daughter.
    assert (2, 6) in g.edges
    assert g.out_degree(2) == 2
    assert g.dividing_nodes() == [2]
    # Non-destructive: no node added/removed, primary edge preserved.
    assert g.num_nodes == 9
    assert (2, 3) in g.edges


def test_recovered_fork_scores_a_division_tp() -> None:
    g = _division_scenario()
    apply_division_overlay(g, SCALE, DEFAULT)
    # Self-score: the recovered strongly-connected fork is a division TP, and the
    # champion (pre-overlay) graph misses it entirely.
    assert division_counts(g, g) == (1, 0, 0)
    assert division_counts(_division_scenario(), g) == (0, 1, 0)


def test_sibling_ratio_gate_rejects_a_far_second_daughter() -> None:
    g = _division_scenario()
    # Move the second-daughter head far from the parent relative to the primary
    # (d1 = 1 µm, d2 = 4 µm): ratio 2.0 rejects it.
    for nid, (t, z, y, x) in list(g.coords.items()):
        if nid in (6, 7, 8):
            g.coords[nid] = (t, z, 4.0, x)
    apply_division_overlay(g, SCALE, ("nearest-head", 7.0, 2.0, 2, True))
    assert g.out_degree(2) == 1  # no fork
    assert g.dividing_nodes() == []


def test_min_daughter_len_gate_rejects_a_transient_head() -> None:
    g = _division_scenario()
    # Truncate the second daughter to a single node (no successor) -> length 1.
    g = g.subgraph([n for n in g.node_ids() if n not in (7, 8)])
    apply_division_overlay(g, SCALE, ("nearest-head", 7.0, 2.0, 2, True))
    assert g.out_degree(2) == 1  # transient head is not a credible daughter


def test_require_parent_track_rejects_a_rootless_parent() -> None:
    g = _division_scenario()
    # Drop the parent's incoming edges so P2 starts its own track.
    g = g.subgraph([n for n in g.node_ids() if n not in (0, 1)])
    apply_division_overlay(g, SCALE, ("nearest-head", 7.0, 2.0, 2, True))
    assert g.out_degree(2) == 1  # a track-start parent cannot divide
    # ...but allowing rootless parents re-enables it.
    g2 = _division_scenario().subgraph(
        [n for n in _division_scenario().node_ids() if n not in (0, 1)]
    )
    apply_division_overlay(g2, SCALE, ("nearest-head", 7.0, 2.0, 2, False))
    assert g2.out_degree(2) == 2


def test_each_head_consumed_by_at_most_one_parent() -> None:
    # Two candidate parents at t=1, each with a primary daughter chain, and ONE
    # shared head at t=2 between them: the head must attach to exactly one parent.
    g = TrackingGraph.from_lists(
        {
            0: (0, 0, -2, 0), 1: (1, 0, -2, 0),      # parent A: 0->1
            2: (2, 0, -2, 0), 3: (3, 0, -2, 0),      # A primary chain 1->2->3
            4: (0, 0, 2, 0), 5: (1, 0, 2, 0),        # parent B: 4->5
            6: (2, 0, 2, 0), 7: (3, 0, 2, 0),        # B primary chain 5->6->7
            8: (2, 0, 0, 0), 9: (3, 0, 0, 0),        # shared head at y=0
        },
        [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (8, 9)],
    )
    apply_division_overlay(g, SCALE, ("nearest-head", 7.0, 0.0, 2, True))
    consumers = [p for p in (1, 5) if (p, 8) in g.edges]
    assert len(consumers) == 1  # head 8 attached to exactly one parent


def test_link_centroids_default_leaves_overlay_off() -> None:
    # A tiny two-frame detection set; default LinkParams must not divide.
    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
    }
    g_default = link_centroids(dets, scale=SCALE, params=LinkParams())
    g_explicit_off = link_centroids(
        dets, scale=SCALE, params=LinkParams(division_overlay=None)
    )
    assert g_default.edges == g_explicit_off.edges
    assert LinkParams().division_overlay is None


# --- SOT-2898: mutual-NN precision fork ------------------------------------
# ("mutual-nn", max_distance, sibling_ratio, min_daughter_len,
#  require_parent_track, require_primary_persist, mutual_margin)
MUTUAL = ("mutual-nn", 7.0, 2.0, 2, True, True, 0.0)


def test_mutual_nn_off_is_a_no_op_byte_for_byte() -> None:
    g = _division_scenario()
    before_edges = list(g.edges)
    apply_division_overlay(g, SCALE, None)
    assert g.edges == before_edges


def test_mutual_nn_fires_on_a_clean_unambiguous_split() -> None:
    # The single-parent scenario: P2 is the only cell at t=2, so it is trivially
    # the mutual nearest neighbour of the dropped head, and both daughters persist.
    g = _division_scenario()
    apply_division_overlay(g, SCALE, MUTUAL)
    assert g.out_degree(2) == 2  # dropped daughter re-attached => a 1->2 fork
    assert (2, 6) in g.edges


def test_mutual_nn_rejects_when_primary_daughter_does_not_persist() -> None:
    # Truncate the PRIMARY daughter chain to a single node: a real split keeps
    # BOTH daughters, so require_primary_persist rejects the fork.
    g = _division_scenario().subgraph(
        [n for n in _division_scenario().node_ids() if n not in (4, 5)]
    )
    apply_division_overlay(g, SCALE, MUTUAL)
    assert g.out_degree(2) == 1  # primary dies immediately => not a credible split
    # ...but with require_primary_persist=False the symmetric test is waived.
    g2 = _division_scenario().subgraph(
        [n for n in _division_scenario().node_ids() if n not in (4, 5)]
    )
    apply_division_overlay(
        g2, SCALE, ("mutual-nn", 7.0, 2.0, 2, True, False, 0.0)
    )
    assert g2.out_degree(2) == 2


def test_mutual_nn_rejects_when_a_rival_parent_is_as_close_to_the_head() -> None:
    # Two candidate parents at t=2 and a dropped head sitting BETWEEN them, so
    # neither parent is the head's unambiguous nearest neighbour.
    g = TrackingGraph.from_lists(
        {
            0: (1, 0, -3, 0), 1: (2, 0, -3, 0),   # parent A track 0->1
            2: (3, 0, -3, 0), 3: (4, 0, -3, 0),   # A primary chain 1->2->3
            4: (1, 0, 3, 0), 5: (2, 0, 3, 0),     # parent B track 4->5
            6: (3, 0, 3, 0), 7: (4, 0, 3, 0),     # B primary chain 5->6->7
            8: (3, 0, 0, 0), 9: (4, 0, 0, 0),     # dropped head midway (y=0)
        },
        [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (8, 9)],
    )
    # sibling_ratio=0 disables the sibling-distance gate so only the mutual-NN
    # test decides: head 8 is equidistant (3um) from parents 1 and 5 => ambiguous.
    apply_division_overlay(g, SCALE, ("mutual-nn", 7.0, 0.0, 2, True, True, 0.0))
    assert (1, 8) not in g.edges and (5, 8) not in g.edges  # rejected as ambiguous


def test_champion_params_preserves_mutual_nn_seven_tuple() -> None:
    # The config parser must pass the 6th/7th elements through, not truncate to 5.
    from biohub_tracking.champion import champion_params, load_champion_config

    cfg = load_champion_config()
    cfg = {**cfg, "link": {**cfg["link"], "division_overlay": list(MUTUAL)}}
    _detect, link, _scale = champion_params(cfg)
    assert link.division_overlay == MUTUAL
    assert len(link.division_overlay) == 7


# --- SOT-2932: decoupled split-signature detector overlay ------------------
# ("split-signature", max_distance, sibling_ratio, min_daughter_len,
#  require_parent_track, require_primary_persist, mutual_margin,
#  straddle_max, parent_bright_pct, daughter_bright_pct)
SPLIT_GEOM = ("split-signature", 7.0, 2.0, 2, True, True, 0.0, 1.0, 0.0, 0.0)


def test_split_signature_off_is_a_no_op_byte_for_byte() -> None:
    g = _division_scenario()
    before_edges = list(g.edges)
    out = apply_division_overlay(g, SCALE, None, node_response={})
    assert out is g
    assert g.edges == before_edges


def test_split_signature_fires_on_a_bipolar_split_geometry_only() -> None:
    # The primary (y=-1) and dropped head (y=+1) straddle the parent: |u1+u2|=0,
    # well under straddle_max. With no intensity gate (bright_pct=0) it fires.
    g = _division_scenario()
    apply_division_overlay(g, SCALE, SPLIT_GEOM, node_response=None)
    assert g.out_degree(2) == 2
    assert (2, 6) in g.edges


def test_split_signature_straddle_gate_rejects_a_same_side_head() -> None:
    # Move the second daughter to the SAME side as the primary (both y=-1): the
    # daughters no longer straddle the parent (|u1+u2|~2), so a tight straddle_max
    # rejects the fork even though the distance/sibling gates would accept it.
    g = _division_scenario()
    for nid in (6, 7, 8):
        t, z, _y, x = g.coords[nid]
        g.coords[nid] = (t, z, -1.0, x + 0.5)  # co-linear with the primary daughter
    apply_division_overlay(
        g, SCALE, ("split-signature", 7.0, 2.0, 2, True, True, 0.0, 0.5, 0.0, 0.0)
    )
    assert g.out_degree(2) == 1  # not bipolar -> rejected


def test_split_signature_intensity_gate_rejects_a_dim_daughter() -> None:
    # Bipolar geometry passes, but the dropped head is DIM in its frame (low
    # response percentile), so a positive daughter_bright_pct rejects it.
    g = _division_scenario()
    # Add a bright decoy at t=3 so head 6 ranks at the bottom of its frame.
    g.add_node(20, 3, 0, 5, 0)
    resp = {n: 10.0 for n in g.node_ids()}
    resp[6] = 0.0  # dropped head is the dimmest node at t=3
    params = ("split-signature", 7.0, 2.0, 2, True, True, 0.0, 1.0, 0.0, 0.5)
    apply_division_overlay(g, SCALE, params, node_response=resp)
    assert g.out_degree(2) == 1  # dim head is not a credible daughter blob
    # ...and with the response gate disabled (pct=0) the same head fires.
    g2 = _division_scenario()
    apply_division_overlay(
        g2, SCALE, ("split-signature", 7.0, 2.0, 2, True, True, 0.0, 1.0, 0.0, 0.0),
        node_response=resp,
    )
    assert g2.out_degree(2) == 2


def test_split_signature_none_response_skips_intensity_gate() -> None:
    # A positive bright_pct with node_response=None must NOT crash and must simply
    # skip the intensity gate (pipeline path supplies no image features).
    g = _division_scenario()
    apply_division_overlay(
        g, SCALE, ("split-signature", 7.0, 2.0, 2, True, True, 0.0, 1.0, 0.9, 0.9),
        node_response=None,
    )
    assert g.out_degree(2) == 2  # geometry-only fallback still fires


def test_champion_params_preserves_split_signature_ten_tuple() -> None:
    from biohub_tracking.champion import champion_params, load_champion_config

    cfg = load_champion_config()
    cfg = {**cfg, "link": {**cfg["link"], "division_overlay": list(SPLIT_GEOM)}}
    _detect, link, _scale = champion_params(cfg)
    assert link.division_overlay == SPLIT_GEOM
    assert len(link.division_overlay) == 10
