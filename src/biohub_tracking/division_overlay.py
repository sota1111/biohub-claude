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
    node_response: dict[int, float] | None = None,
) -> TrackingGraph:
    """Add high-precision second-daughter edges to *graph* in place; return it.

    ``params`` is ``None`` (or an empty tuple) to disable — the graph is returned
    untouched, byte-for-byte. The leading element is a ``kind`` tag selecting the
    overlay mechanism (the config shape can grow without breaking older configs):

    * ``("nearest-head", max_distance, sibling_ratio, min_daughter_len,
      require_parent_track)`` — SOT-2818: attach the nearest persistent head to a
      parent whose second daughter the linker dropped.
    * ``("mutual-nn", max_distance, sibling_ratio, min_daughter_len,
      require_parent_track, require_primary_persist, mutual_margin)`` — SOT-2898:
      the SOT-2818 gate PLUS a **mutual-nearest-neighbour** parent test and a
      **symmetric daughter-persistence** test (see :func:`_apply_mutual_nn`).
    * ``("split-signature", max_distance, sibling_ratio, min_daughter_len,
      require_parent_track, require_primary_persist, mutual_margin, straddle_max,
      parent_bright_pct, daughter_bright_pct)`` — SOT-2932: the mutual-nn geometric
      base PLUS a **decoupled image split-signature detector** — a bipolar
      straddle-geometry gate and per-frame intensity-condensation gates evaluated
      from ``node_response`` (see :func:`_apply_split_signature`).

    ``node_response`` maps ``node_id -> detection-response value`` (the champion DoG
    response sampled at the node's voxel, supplied by the caller that read the
    image). It is used ONLY by the ``split-signature`` kind; when it is ``None`` the
    image-intensity gates are skipped (so the linking pipeline path, which never
    reads image features here, still runs the geometric gates only). The champion
    never enables any overlay, so all in-pipeline call sites are unaffected.
    """
    if not params:
        return graph
    kind = str(params[0])
    if kind == "nearest-head":
        return _apply_nearest_head(graph, scale, params)
    if kind == "mutual-nn":
        return _apply_mutual_nn(graph, scale, params)
    if kind == "split-signature":
        return _apply_split_signature(graph, scale, params, node_response)
    raise ValueError(f"unknown division_overlay kind: {kind!r}")


def _apply_nearest_head(
    graph: TrackingGraph,
    scale: tuple[float, float, float],
    params: DivisionOverlayParams,
) -> TrackingGraph:
    """SOT-2818 overlay: nearest persistent head becomes the second daughter."""
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


def _apply_mutual_nn(
    graph: TrackingGraph,
    scale: tuple[float, float, float],
    params: DivisionOverlayParams,
) -> TrackingGraph:
    """SOT-2898 precision-first overlay: mutual-NN + symmetric-persistence fork.

    This shares SOT-2818's non-destructive contract (only *adds* one
    ``parent -> nearby-head`` edge per fork; OFF / zero-fire ⇒ champion graph
    byte-for-byte) but tightens the gate to fire only on **high-confidence** forks,
    to buy division TP without spraying the fork FPs that sank SOT-2762 / SOT-2818
    on the dense ``6bba`` families:

    * **Mutual nearest neighbour (edge-FP explicit guard).** The candidate second
      daughter ``D2`` (an unclaimed persistent head at ``t+1``) is attached to a
      parent ``P`` only if ``P`` is the *nearest candidate parent* to ``D2`` among
      all cells at ``t`` that could adopt it — i.e. ``P`` and ``D2`` are mutual
      nearest neighbours. If some *other* established track sits closer to ``D2``,
      ``D2`` more likely belongs to (or is a decoy near) that track, so attaching it
      to ``P`` would be a division FP *and* an edge FP; the mutual test rejects it.
    * **Unambiguous margin.** With ``mutual_margin > 0`` the runner-up parent must be
      at least ``(1 + mutual_margin)`` times farther than ``P`` — the split must be
      unambiguous, not a near-tie between two possible parents.
    * **Symmetric daughter persistence (±1-window consistency).** The division
      metric scores a fork inside a ``parent -> divider -> children -> grandchildren``
      window, so a real split has *both* daughters continue. With
      ``require_primary_persist`` the champion-assigned primary daughter ``C`` must
      also reach ``min_daughter_len`` nodes, not just ``D2`` — a fork where one
      "daughter" dies immediately is rejected.

    Params:
    ``("mutual-nn", max_distance, sibling_ratio, min_daughter_len,
    require_parent_track, require_primary_persist, mutual_margin)``.
    """
    max_distance = float(params[1])
    sibling_ratio = float(params[2])
    min_daughter_len = int(params[3])
    require_parent_track = bool(params[4])
    require_primary_persist = bool(params[5]) if len(params) > 5 else True
    mutual_margin = float(params[6]) if len(params) > 6 else 0.0

    scale_arr = np.asarray(scale, dtype=float)
    nodes_by_time = graph.nodes_by_time()

    # Persistent unclaimed heads per timepoint (a credible daughter lineage): no
    # predecessor and a forward track of >= min_daughter_len nodes.
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

    # Candidate parents per timepoint: exactly one champion successor (so adding one
    # edge makes a legal 1->2 fork) and — when required — a predecessor.
    parents_by_time: dict[int, list[int]] = {}
    for t, ids in nodes_by_time.items():
        ps = [
            n
            for n in ids
            if graph.out_degree(n) == 1
            and not (require_parent_track and graph.in_degree(n) == 0)
        ]
        if ps:
            parents_by_time[t] = sorted(ps)

    def scaled_dist(a: int, b: int) -> float:
        pa = graph.position(a) * scale_arr
        pb = graph.position(b) * scale_arr
        return float(np.sqrt(((pa - pb) ** 2).sum()))

    consumed: set[int] = set()  # a head becomes at most one parent's daughter
    additions: list[tuple[int, int]] = []

    # Deterministic parent order (ascending id).
    for parent in sorted(pid for pids in parents_by_time.values() for pid in pids):
        t_parent = graph.t(parent)
        heads = heads_by_time.get(t_parent + 1)
        if not heads:
            continue

        primary = graph.successors(parent)[0]
        if require_primary_persist and (
            _forward_track_length(graph, primary, min_daughter_len)
            < min_daughter_len
        ):
            continue  # a real split keeps BOTH daughters alive
        d1 = scaled_dist(parent, primary)

        # Nearest eligible head to this parent (P's side of the mutual test).
        best: tuple[float, int] | None = None  # (d2, head id)
        for h in heads:
            if h in consumed or h == primary:
                continue
            d2 = scaled_dist(parent, h)
            if d2 > max_distance:
                continue
            if sibling_ratio > 0.0 and d2 > sibling_ratio * d1:
                continue  # sibling too far vs the primary daughter -> not a split
            if best is None or (d2, h) < best:
                best = (d2, h)
        if best is None:
            continue
        d2, head = best

        # Mutual-NN test: P must be the nearest candidate parent to `head`, by a
        # clear margin. Scan sibling candidate parents at the same timepoint.
        rivals = parents_by_time.get(t_parent, ())
        mutual = True
        for other in rivals:
            if other == parent:
                continue
            do = scaled_dist(other, head)
            if do <= d2 * (1.0 + mutual_margin):
                mutual = False  # another parent is as close (or closer) -> ambiguous
                break
        if not mutual:
            continue

        consumed.add(head)
        additions.append((parent, head))

    for parent, head in additions:
        graph.add_edge(parent, head)
    return graph


def _percentile_rank_by_time(
    graph: TrackingGraph,
    node_response: dict[int, float],
) -> dict[int, float]:
    """Map each node to the percentile rank (0..1) of its response WITHIN its frame.

    A true mitotic parent is transiently the brightest thing in its neighbourhood
    (chromatin condensation / rounding), so an *absolute* response threshold would
    be swamped by the strong intensity drift across the developing embryo. Ranking
    each node against the other detections at the *same* timepoint makes the
    brightness gate self-calibrating per frame — the same idea the champion's
    ``mad_k`` robust-z detection threshold uses, applied here to the already-linked
    nodes. Nodes with no recorded response rank 0.0 (fail-closed: they cannot pass a
    positive brightness gate).
    """
    ranks: dict[int, float] = {}
    for _t, ids in graph.nodes_by_time().items():
        vals = [(node_response.get(n, float("-inf")), n) for n in ids]
        vals.sort()
        m = len(vals)
        denom = float(m - 1) if m > 1 else 1.0
        for i, (_v, n) in enumerate(vals):
            ranks[n] = i / denom
    return ranks


def _apply_split_signature(
    graph: TrackingGraph,
    scale: tuple[float, float, float],
    params: DivisionOverlayParams,
    node_response: dict[int, float] | None,
) -> TrackingGraph:
    """SOT-2932 decoupled division-event detector overlay (image split-signature).

    **Structural difference vs SOT-2818 / SOT-2762 / SOT-2898 (new formulation).**
    Every prior division attempt is either *inside the linking cost* (SOT-2762
    ``allow_division`` re-runs the LAP assignment) or a *graph-geometry-only*
    post-processing overlay (SOT-2818 nearest-head, SOT-2898 mutual-nn) that
    re-attaches a nearby persistent head as the second daughter using **positions
    alone**. Those overlays were rejected because the head they pick is usually
    *not* the true daughter, so the added ``P -> D2`` edge is a division FP *and* an
    edge FP, and the edge component regresses (SOT-2898: even the tightest gate that
    earned ``div_tp=1`` still added ``>=4`` edge FPs and regressed ``micro_adj``).

    This overlay keeps the non-destructive contract (only *adds* one
    ``parent -> head`` edge per fork; OFF / zero-fire ⇒ champion graph byte-for-byte)
    and the mutual-nn geometric base, but proposes a division ONLY when a **local
    strong split signature** — image intensity + bipolar daughter geometry, fully
    decoupled from the linking LAP — is present, driving fork FP toward zero:

    * **Bipolar straddle geometry (shape).** A real mitotic split is bipolar: the
      two daughters separate to roughly *opposite* sides of the parent. With unit
      vectors ``u1 = (C - P)/|C - P|`` and ``u2 = (D2 - P)/|D2 - P|`` the straddle
      ``|u1 + u2|`` is ~0 for opposed daughters and ~2 for co-linear ones; require
      ``|u1 + u2| <= straddle_max``. The mutual-nn overlay never tested this — it
      accepts a head sitting on the *same* side as the primary daughter (a track
      passing by), the dominant FP mode.
    * **Parent condensation (intensity).** ``P`` must be a bright node in its frame
      (per-frame response percentile ``>= parent_bright_pct``): a dividing nucleus
      condenses and brightens just before cytokinesis.
    * **Both-daughter blobness (intensity).** ``C`` and ``D2`` must each be genuine
      blobs (per-frame percentile ``>= daughter_bright_pct``), so a dim decoy head
      that merely happens to straddle the parent is rejected.

    When ``node_response is None`` the two intensity gates are skipped and only the
    geometric (mutual-nn + straddle) gates apply — but the champion never enables
    this overlay, so the in-pipeline call sites are byte-for-byte unchanged.

    Params:
    ``("split-signature", max_distance, sibling_ratio, min_daughter_len,
    require_parent_track, require_primary_persist, mutual_margin, straddle_max,
    parent_bright_pct, daughter_bright_pct)``.
    """
    max_distance = float(params[1])
    sibling_ratio = float(params[2])
    min_daughter_len = int(params[3])
    require_parent_track = bool(params[4])
    require_primary_persist = bool(params[5]) if len(params) > 5 else True
    mutual_margin = float(params[6]) if len(params) > 6 else 0.0
    straddle_max = float(params[7]) if len(params) > 7 else 2.0
    parent_bright_pct = float(params[8]) if len(params) > 8 else 0.0
    daughter_bright_pct = float(params[9]) if len(params) > 9 else 0.0

    scale_arr = np.asarray(scale, dtype=float)
    nodes_by_time = graph.nodes_by_time()

    # Per-frame brightness rank (0..1). Only computed when image response is
    # supplied AND a positive brightness gate is actually requested.
    use_intensity = node_response is not None and (
        parent_bright_pct > 0.0 or daughter_bright_pct > 0.0
    )
    rank = (
        _percentile_rank_by_time(graph, node_response)  # type: ignore[arg-type]
        if use_intensity
        else {}
    )

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

    parents_by_time: dict[int, list[int]] = {}
    for t, ids in nodes_by_time.items():
        ps = [
            n
            for n in ids
            if graph.out_degree(n) == 1
            and not (require_parent_track and graph.in_degree(n) == 0)
        ]
        if ps:
            parents_by_time[t] = sorted(ps)

    def pos(n: int) -> np.ndarray:
        return graph.position(n) * scale_arr

    def scaled_dist(a: int, b: int) -> float:
        d = pos(a) - pos(b)
        return float(np.sqrt((d * d).sum()))

    consumed: set[int] = set()
    additions: list[tuple[int, int]] = []

    for parent in sorted(pid for pids in parents_by_time.values() for pid in pids):
        # Parent condensation gate (intensity).
        if use_intensity and rank.get(parent, 0.0) < parent_bright_pct:
            continue
        t_parent = graph.t(parent)
        heads = heads_by_time.get(t_parent + 1)
        if not heads:
            continue

        primary = graph.successors(parent)[0]
        if require_primary_persist and (
            _forward_track_length(graph, primary, min_daughter_len)
            < min_daughter_len
        ):
            continue
        if use_intensity and rank.get(primary, 0.0) < daughter_bright_pct:
            continue
        d1 = scaled_dist(parent, primary)
        if d1 <= 0.0:
            continue
        p_pos = pos(parent)
        u1 = (pos(primary) - p_pos) / d1

        # Best straddling, bright, mutual-NN head.
        best: tuple[float, int] | None = None  # (d2, head id)
        for h in heads:
            if h in consumed or h == primary:
                continue
            d2 = scaled_dist(parent, h)
            if d2 <= 0.0 or d2 > max_distance:
                continue
            if sibling_ratio > 0.0 and d2 > sibling_ratio * d1:
                continue
            # Bipolar straddle (shape): daughters must separate to opposite sides.
            u2 = (pos(h) - p_pos) / d2
            if float(np.sqrt(((u1 + u2) ** 2).sum())) > straddle_max:
                continue
            # Both-daughter blobness (intensity).
            if use_intensity and rank.get(h, 0.0) < daughter_bright_pct:
                continue
            if best is None or (d2, h) < best:
                best = (d2, h)
        if best is None:
            continue
        d2, head = best

        # Mutual-NN test: P is the nearest candidate parent to `head`, by margin.
        rivals = parents_by_time.get(t_parent, ())
        mutual = True
        for other in rivals:
            if other == parent:
                continue
            if scaled_dist(other, head) <= d2 * (1.0 + mutual_margin):
                mutual = False
                break
        if not mutual:
            continue

        consumed.add(head)
        additions.append((parent, head))

    for parent, head in additions:
        graph.add_edge(parent, head)
    return graph
