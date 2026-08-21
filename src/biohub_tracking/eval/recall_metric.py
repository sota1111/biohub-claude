"""GT-node recall @7 µm — the objective for SOT-2873.

The competition edge metric scores an edge TP only when **both** endpoints match
a GT node within 7 µm (:mod:`biohub_tracking.eval.edge_metric`), so every missed
GT-node detection strands the edges incident to it as FNs. This module isolates
the **coverage** side of that: after the same optimal-bipartite 7 µm node matching
the metric uses (:func:`biohub_tracking.matching.match_nodes`), how many GT nodes
were matched by *some* prediction.

Two coverage numbers are reported because they answer different questions:

* ``gt_node_recall`` — matched GT nodes / all GT nodes. The raw detection
  coverage.
* ``gt_edge_endpoint_recall`` — matched GT nodes / GT nodes that are an endpoint
  of at least one GT edge. This is the quantity the edge metric actually cares
  about: an isolated GT node (no incident GT edge) can never contribute an edge
  TP, so covering it does nothing for the score, whereas covering an
  edge-endpoint GT node is a *necessary* condition for recovering its FN edge.

Pure function of two :class:`~biohub_tracking.graph.TrackingGraph` objects (no
disk, no data), so it unit-tests without the gitignored competition volumes and
is safe to import inside the CV harness.
"""

from __future__ import annotations

from typing import NamedTuple

from ..graph import TrackingGraph
from ..matching import DEFAULT_MAX_DISTANCE, match_nodes


class NodeRecall(NamedTuple):
    """GT-node coverage at the metric's 7 µm match radius."""

    gt_nodes: int
    gt_nodes_matched: int
    gt_node_recall: float
    gt_edge_endpoint_nodes: int
    gt_edge_endpoint_matched: int
    gt_edge_endpoint_recall: float


def _endpoint_nodes(gt: TrackingGraph) -> set[int]:
    """GT node ids that are an endpoint of at least one GT edge."""
    endpoints: set[int] = set()
    for s, t in gt.edges:
        endpoints.add(s)
        endpoints.add(t)
    return endpoints


def gt_node_recall(
    pred: TrackingGraph,
    gt: TrackingGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> NodeRecall:
    """GT-node recall @ ``max_distance`` µm for one (pred, gt) pair.

    Uses the identical optimal-bipartite per-timepoint 7 µm matching the edge
    metric uses, so the matched-GT set here is exactly the set of GT nodes the
    edge metric could see. ``gt_node_recall`` is the fraction of all GT nodes
    matched; ``gt_edge_endpoint_recall`` restricts the denominator to GT nodes
    incident to a GT edge (the only ones that can turn into an edge TP).
    """
    matching = match_nodes(pred, gt, scale=scale, max_distance=max_distance)
    matched_gt = set(matching.values())  # gt ids covered by some prediction

    gt_nodes = gt.num_nodes
    matched = len(matched_gt)
    node_recall = matched / gt_nodes if gt_nodes > 0 else float("nan")

    endpoints = _endpoint_nodes(gt)
    ep_matched = len(endpoints & matched_gt)
    ep_recall = (
        ep_matched / len(endpoints) if endpoints else float("nan")
    )
    return NodeRecall(
        gt_nodes=gt_nodes,
        gt_nodes_matched=matched,
        gt_node_recall=node_recall,
        gt_edge_endpoint_nodes=len(endpoints),
        gt_edge_endpoint_matched=ep_matched,
        gt_edge_endpoint_recall=ep_recall,
    )
