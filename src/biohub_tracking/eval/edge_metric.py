"""Edge Jaccard counting.

Reproduces the competition's edge metric on top of a precomputed node matching:

* A predicted edge is a **true positive (TP)** when both endpoints match GT nodes
  that are themselves connected by a GT edge.
* Every GT edge without such a match is a **false negative (FN)**.
* A predicted edge that is not a TP is a **false positive (FP)** only when it is
  "evaluable" against the sparse ground truth — its source matches a GT node with
  an outgoing GT edge, or its target matches a GT node with an incoming GT edge.
  Predicted edges whose endpoints fall outside the annotated region are ignored
  (the ground truth is sparse, so unmatched predictions are not penalised here;
  the node-count penalty in :mod:`biohub_tracking.eval.score` handles that).

The edge Jaccard is ``TP / (TP + FP + FN)``. Before counting, predicted edges are
sanitised exactly as the reference does: keep only consecutive-frame forward
edges, drop exact duplicates, collapse merges onto a single GT edge, and cap each
node's out-degree at two (a dividing cell has at most two daughters).
"""

from __future__ import annotations

from typing import NamedTuple

from ..graph import TrackingGraph


class EdgeCounts(NamedTuple):
    tp: int
    fp: int
    fn: int


def _sanitise_pred_edges(
    pred: TrackingGraph,
    matching: dict[int, int],
    gt_edge_set: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return the predicted edges kept for scoring, in a deterministic order.

    Edge ids are the edges' insertion index, so all tie-breaks keep the
    lowest-id edge, matching the reference implementation.
    """
    indexed = list(enumerate(pred.edges))  # (edge_id, (src, tgt))

    # 1. Keep only forward, single-step edges: t_target == t_source + 1.
    indexed = [
        (eid, (s, t)) for eid, (s, t) in indexed if pred.t(t) - pred.t(s) == 1
    ]

    # 2. Drop exact duplicate (source, target) pairs, keeping the lowest id.
    seen: set[tuple[int, int]] = set()
    deduped: list[tuple[int, tuple[int, int]]] = []
    for eid, (s, t) in indexed:
        if (s, t) in seen:
            continue
        seen.add((s, t))
        deduped.append((eid, (s, t)))
    indexed = deduped

    # 3. Collapse merges: several predicted edges can map onto the same GT edge
    #    when their endpoints match the same GT nodes. Keep the lowest id per
    #    matched (gt_source, gt_target) pair.
    best_for_gt_edge: dict[tuple[int, int], int] = {}
    for eid, (s, t) in indexed:
        gs, gt_ = matching.get(s), matching.get(t)
        if gs is not None and gt_ is not None:
            key = (gs, gt_)
            if key not in best_for_gt_edge or eid < best_for_gt_edge[key]:
                best_for_gt_edge[key] = eid
    kept: list[tuple[int, tuple[int, int]]] = []
    for eid, (s, t) in indexed:
        gs, gt_ = matching.get(s), matching.get(t)
        if gs is not None and gt_ is not None:
            if best_for_gt_edge[(gs, gt_)] != eid:
                continue  # a lower-id edge already represents this GT edge
        kept.append((eid, (s, t)))
    indexed = kept

    # 4. Cap out-degree at two per source, keeping the two lowest edge ids.
    out_rank: dict[int, int] = {}
    capped: list[tuple[int, tuple[int, int]]] = []
    for eid, (s, t) in sorted(indexed):  # sorted by edge id
        rank = out_rank.get(s, 0) + 1
        out_rank[s] = rank
        if rank <= 2:
            capped.append((eid, (s, t)))

    return [edge for _eid, edge in capped]


def edge_counts(
    pred: TrackingGraph,
    gt: TrackingGraph,
    matching: dict[int, int],
) -> EdgeCounts:
    """Compute edge TP / FP / FN for a matched (pred, gt) pair."""
    gt_edge_set = set(gt.edges)
    gt_out = {n: gt.out_degree(n) for n in gt.node_ids()}
    gt_in = {n: gt.in_degree(n) for n in gt.node_ids()}

    if pred.num_edges == 0 or pred.num_nodes == 0:
        # No predicted structure: every GT edge is a miss.
        return EdgeCounts(tp=0, fp=0, fn=gt.num_edges)

    edges = _sanitise_pred_edges(pred, matching, gt_edge_set)

    tp = 0
    n_valid = 0
    for s, t in edges:
        gs, gt_ = matching.get(s), matching.get(t)
        is_tp = gs is not None and gt_ is not None and (gs, gt_) in gt_edge_set
        out_valid = gs is not None and gt_out.get(gs, 0) > 0
        in_valid = gt_ is not None and gt_in.get(gt_, 0) > 0
        pred_valid = out_valid or in_valid
        if is_tp:
            tp += 1
        if pred_valid:
            n_valid += 1

    fp = n_valid - tp
    fn = gt.num_edges - tp
    return EdgeCounts(tp=tp, fp=fp, fn=fn)
