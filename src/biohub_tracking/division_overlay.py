"""Non-destructive division-event overlay (SOT-2818).

The champion ``detect-link-dog-v4-shorttrack`` runs ``link.allow_division=False``
and therefore forfeits the official metric's ``0.1 · division_jaccard`` term: it
predicts zero forks, so every ground-truth division is a division FN and the
division Jaccard is 0.0. Turning division ON *inside* the linker (SOT-2762) was
rejected — enabling ``allow_division`` re-runs the LAP assignment, which reassigns
leftover detections and re-picks the primary daughter, spraying fork FPs on the
dense ``6bba`` families and losing edge TP across all 16 variants.

This module recovers the 0.1 term by a **different mechanism** that never touches
the linking assignment. It is a pure *post-processing overlay* applied to the
champion's already-linked, already-pruned :class:`~biohub_tracking.graph.TrackingGraph`:

* **What a real division looks like in champion output.** At a genuine split the
  parent cell at ``t`` has two daughters ``(d_a, d_b)`` at ``t+1``. The champion's
  one-to-one optimal assignment can attach the parent to only *one* of them (say
  ``d_a``), so ``d_b`` is left unmatched and becomes a fresh **head** (a node with
  no predecessor) that starts its own track. The champion therefore already
  encodes half of every division — it only *drops the second daughter edge*.
* **The overlay re-attaches that dropped daughter, and nothing else.** For a
  parent ``P`` that (a) sits on an established track (has a predecessor) and (b)
  has exactly its one champion-assigned primary daughter ``C``, the overlay looks
  for a nearby persistent head ``D2`` at ``t+1`` and *adds* the single edge
  ``P -> D2`` — turning ``P`` into a ``1 -> 2`` fork. The primary edge ``P -> C``
  is never moved, no node is added or removed, and no existing edge is deleted, so
  when the overlay is OFF (or fires zero times) the champion graph is byte-for-byte
  unchanged.

**Why this is high precision (and how it protects the edge metric).** Every added
edge is scored by the edge metric too, so a spurious second daughter is both a
division FP *and* an edge FP. The overlay is therefore gated hard:

* ``max_distance`` — ``P -> D2`` scaled distance ``<= max_distance`` µm (the same
  7 µm match radius; a division's daughters are born adjacent to the parent).
* ``sibling_ratio`` — ``d2 <= sibling_ratio · d1`` where ``d1`` is the scaled
  ``P -> C`` distance: a real mitotic split yields two daughters roughly
  symmetric about the parent, so a candidate much farther than the primary
  daughter is rejected (the dominant over-split failure mode on dense volumes).
* ``min_daughter_len`` — ``D2`` must *persist*: its forward track (following
  successors from the head) must reach at least this many nodes, so a transient
  one-frame decoy cannot masquerade as a daughter lineage.
* ``require_parent_track`` — ``P`` must have a predecessor, so only cells on an
  established lineage can divide (a track that just started is not a credible
  dividing parent).

The parent must have out-degree exactly 1 before the overlay (so adding one edge
makes a legal ``1 -> 2`` fork, never a ``1 -> 3`` over-split), each head ``D2`` is
consumed by at most one parent, and each parent gains at most one second daughter.
Ties are broken deterministically (nearest ``D2`` by scaled distance, then lowest
node id), so the overlay is a deterministic same-seed transform — an A/B against
the champion is a clean single-variable ablation.
"""

from __future__ import annotations

import numpy as np

from .graph import TrackingGraph

# A division-overlay configuration, kept as a plain tuple so it round-trips
# through ``champion/config.json`` as data (mirrors how detect knobs are stored).
# ``("nearest-head", max_distance, sibling_ratio, min_daughter_len, require_parent_track)``.
DivisionOverlayParams = tuple


def _forward_track_length(graph: TrackingGraph, head: int, cap: int) -> int:
    """Number of nodes reachable forward from *head* along successors, up to *cap*.

    With ``allow_division=False`` the champion graph is a forest of simple chains,
    so following the (single) successor measures the head's track length. Capped so
    a long track short-circuits the walk once the persistence gate is satisfied.
    """
    count = 1
    node = head
    seen = {head}
    while count < cap:
        succ = graph.successors(node)
        if not succ:
            break
        nxt = succ[0]
        if nxt in seen:  # defensive: never loop on a malformed cycle
            break
        seen.add(nxt)
        node = nxt
        count += 1
    return count


def apply_division_overlay(
    graph: TrackingGraph,
    scale: tuple[float, float, float],
    params: DivisionOverlayParams | None,
) -> TrackingGraph:
    """Add high-precision second-daughter edges to *graph* in place; return it.

    ``params`` is ``None`` (or an empty tuple) to disable — the graph is returned
    untouched, byte-for-byte. Otherwise it is
    ``(kind, max_distance, sibling_ratio, min_daughter_len, require_parent_track)``
    with ``kind == "nearest-head"`` (the only mechanism today; kept as a leading
    tag so future overlay variants can be added without changing the config shape).
    """
    if not params:
        return graph
    kind = str(params[0])
    if kind != "nearest-head":
        raise ValueError(f"unknown division_overlay kind: {kind!r}")
    max_distance = float(params[1])
    sibling_ratio = float(params[2])
    min_daughter_len = int(params[3])
    require_parent_track = bool(params[4])

    scale_arr = np.asarray(scale, dtype=float)
    nodes_by_time = graph.nodes_by_time()

    # Candidate heads per timepoint: nodes with no predecessor that persist for at
    # least ``min_daughter_len`` frames (a credible daughter lineage). Precomputed
    # so each parent lookup is a cheap dict hit.
    heads_by_time: dict[int, list[int]] = {}
    for t, ids in nodes_by_time.items():
        heads = [
            n
            for n in ids
            if graph.in_degree(n) == 0
            and _forward_track_length(graph, n, min_daughter_len) >= min_daughter_len
        ]
        if heads:
            heads_by_time[t] = sorted(heads)

    consumed: set[int] = set()  # a head may become at most one parent's daughter
    additions: list[tuple[int, int]] = []

    # Parents in ascending id order for determinism. A candidate parent has exactly
    # one successor (its champion primary daughter) and — when required — a
    # predecessor (sits on an established track).
    for parent in sorted(graph.node_ids()):
        if graph.out_degree(parent) != 1:
            continue
        if require_parent_track and graph.in_degree(parent) == 0:
            continue
        t_parent = graph.t(parent)
        heads = heads_by_time.get(t_parent + 1)
        if not heads:
            continue

        primary = graph.successors(parent)[0]
        p_pos = graph.position(parent) * scale_arr
        c_pos = graph.position(primary) * scale_arr
        d1 = float(np.sqrt(((p_pos - c_pos) ** 2).sum()))

        best: tuple[float, int] | None = None  # (d2, head id)
        for h in heads:
            if h in consumed or h == primary:
                continue
            h_pos = graph.position(h) * scale_arr
            d2 = float(np.sqrt(((p_pos - h_pos) ** 2).sum()))
            if d2 > max_distance:
                continue
            if sibling_ratio > 0.0 and d2 > sibling_ratio * d1:
                continue  # sibling too far vs primary daughter → not a split
            if best is None or (d2, h) < best:
                best = (d2, h)

        if best is not None:
            head = best[1]
            consumed.add(head)
            additions.append((parent, head))

    for parent, head in additions:
        graph.add_edge(parent, head)
    return graph
