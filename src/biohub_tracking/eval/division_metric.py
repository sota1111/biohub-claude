"""Division Jaccard counting.

A ground-truth **cell division** is a node with exactly two outgoing edges; any
predicted node with at least two outgoing edges is treated as a predicted *fork*.
Because the exact frame at which a cell visibly splits is subjective, each GT
division is evaluated inside a local window::

    grandparent -> dividing parent -> children -> grandchildren

and a predicted fork may occur one timepoint before or after the GT split. This
module is a clean-room port of the reference scoring rules
(``royerlab/kaggle-cell-tracking-competition``, BSD-3-Clause) onto the
dependency-light :class:`~biohub_tracking.graph.TrackingGraph`.

The division Jaccard is ``TP / (TP + FP + FN)``.
"""

from __future__ import annotations

from typing import NamedTuple

from ..graph import TrackingGraph
from ..matching import match_nodes


class DivisionCounts(NamedTuple):
    tp: int
    fn: int
    fp: int


class DivisionScores(NamedTuple):
    scores: dict[int, int]  # GT divider id -> 1 (recovered) or 0
    tp_forks: set[int]
    fp_forks: set[int]


def extract_divisions(graph: TrackingGraph) -> dict[int, TrackingGraph]:
    """Extract each GT division event as a local subgraph.

    The window is ``parent -> divider -> children -> grandchildren``.
    """
    divisions: dict[int, TrackingGraph] = {}
    for div_node in graph.dividing_nodes():
        parents = graph.predecessors(div_node)
        children = graph.successors(div_node)
        grandchildren = [gc for child in children for gc in graph.successors(child)]
        keep = [*parents, div_node, *children, *grandchildren]
        divisions[div_node] = graph.subgraph(keep)
    return divisions


def _bipartite_max_matching(
    left: list[int],
    edges: dict[int, set[int]],
) -> dict[int, int]:
    """Maximum-cardinality bipartite matching via DFS augmenting paths."""
    match_r: dict[int, int] = {}
    match_l: dict[int, int] = {}

    def augment(u: int, seen: set[int]) -> bool:
        for v in edges.get(u, ()):
            if v in seen:
                continue
            seen.add(v)
            if v not in match_r or augment(match_r[v], seen):
                match_l[u] = v
                match_r[v] = u
                return True
        return False

    for u in left:
        augment(u, set())
    return match_l


def _matched_division_nodes(
    matching: dict[int, int],
    gt_div: TrackingGraph,
    divider_id: int,
) -> tuple[set[int], list[set[int]]] | None:
    """Group matched pred nodes into a parent side and daughter lineages."""
    if not matching:
        return None

    node_to_gt = dict(matching)
    gt_children = gt_div.successors(divider_id)
    if len(gt_children) < 2:
        return None

    gt_parent_ids = {divider_id, *gt_div.predecessors(divider_id)}
    parent_ids = {pred_id for pred_id, gt_id in node_to_gt.items() if gt_id in gt_parent_ids}
    daughter_ids = [
        {pred_id for pred_id, gt_id in node_to_gt.items() if gt_id in {child, *gt_div.successors(child)}}
        for child in gt_children
    ]
    if not parent_ids or sum(bool(ids) for ids in daughter_ids) < 2:
        return None
    return parent_ids, daughter_ids


def _is_strongly_connected_division(
    pred: TrackingGraph,
    pred_div: int,
    parent_ids: set[int],
    daughter_ids: list[set[int]],
) -> bool:
    """Check a predicted fork's local directed topology."""
    pred_parent_ids = {pred_div, *pred.predecessors(pred_div)}
    if pred_parent_ids.isdisjoint(parent_ids):
        return False

    pred_lineages = [{child, *pred.successors(child)} for child in pred.successors(pred_div)]
    lineage_edges = {
        gt_lineage: {
            pred_lineage
            for pred_lineage, pred_ids in enumerate(pred_lineages)
            if not matched_ids.isdisjoint(pred_ids)
        }
        for gt_lineage, matched_ids in enumerate(daughter_ids)
    }
    return len(_bipartite_max_matching(list(lineage_edges), lineage_edges)) >= 2


def _gt_weak_component_ids(graph: TrackingGraph) -> dict[int, int]:
    """Map each GT node to its weakly connected component id."""
    component_ids: dict[int, int] = {}
    for seed in graph.node_ids():
        if seed in component_ids:
            continue
        component_ids[seed] = seed
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in graph.successors(current) + graph.predecessors(current):
                if neighbor not in component_ids:
                    component_ids[neighbor] = seed
                    stack.append(neighbor)
    return component_ids


def _branch_component_evidence(
    pred: TrackingGraph,
    pred_div: int,
    child: int,
    pred_to_gt: dict[int, int],
    gt_component: dict[int, int],
) -> tuple[int | None, bool]:
    """Return one GT component for a predicted child branch.

    Direct-child evidence takes precedence over grandchildren. The boolean flags
    a locally merged branch that cannot be assigned uniquely to this fork.
    """
    if set(pred.predecessors(child)) != {pred_div}:
        return None, True
    if child in pred_to_gt:
        return gt_component[pred_to_gt[child]], False

    grandchildren = pred.successors(child)
    if any(set(pred.predecessors(node)) != {child} for node in grandchildren):
        return None, True

    components = {
        gt_component[pred_to_gt[node]] for node in grandchildren if node in pred_to_gt
    }
    if len(components) == 1:
        return next(iter(components)), False
    return None, False


def _pred_division_fork_sets(
    pred: TrackingGraph,
    gt: TrackingGraph,
    scale: tuple[float, ...] | None,
    max_distance: float,
) -> tuple[set[int], set[int], set[int]]:
    """Return evaluable, cross-component, and malformed predicted forks."""
    pred_to_gt = match_nodes(pred, gt, scale=scale, max_distance=max_distance)

    pred_forks = {n for n in pred.node_ids() if pred.out_degree(n) >= 2}
    evaluable_forks = {
        pred_id
        for pred_id in pred_forks
        if pred_id in pred_to_gt and gt.out_degree(pred_to_gt[pred_id]) >= 1
    }

    gt_component = _gt_weak_component_ids(gt)
    cross_component_forks: set[int] = set()
    malformed_forks: set[int] = set()
    for pred_id in pred_forks:
        branch_evidence: list[int] = []
        for child in pred.successors(pred_id):
            component, malformed = _branch_component_evidence(
                pred, pred_id, child, pred_to_gt, gt_component
            )
            if malformed:
                malformed_forks.add(pred_id)
                break
            if component is not None:
                branch_evidence.append(component)
        else:
            if len(set(branch_evidence)) >= 2:
                cross_component_forks.add(pred_id)

    return evaluable_forks, cross_component_forks, malformed_forks


def score_divisions(
    pred: TrackingGraph,
    gt: TrackingGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> DivisionScores:
    """Score each GT division: 1 if recovered, 0 otherwise, plus fork classes."""
    gt_divisions = extract_divisions(gt)
    matched: dict[int, dict[int, int]] = {
        div_node: match_nodes(
            pred, gt, scale=scale, max_distance=max_distance,
            gt_node_subset=gt_div.node_ids(),
        )
        for div_node, gt_div in gt_divisions.items()
    }

    pred_div_nodes = {n for n in pred.node_ids() if pred.out_degree(n) >= 2}
    evaluable_forks, cross_component_forks, malformed_forks = _pred_division_fork_sets(
        pred, gt, scale, max_distance
    )
    invalid_forks = cross_component_forks | malformed_forks

    candidates: dict[int, set[int]] = {}
    considered: set[int] = set()
    for div_node, matching in matched.items():
        matched_nodes = _matched_division_nodes(matching, gt_divisions[div_node], div_node)
        if matched_nodes is None:
            candidates[div_node] = set()
            continue

        parent_ids, daughter_ids = matched_nodes
        local_nodes = parent_ids | {
            successor for parent_id in parent_ids for successor in pred.successors(parent_id)
        }
        local_forks = local_nodes & pred_div_nodes
        considered |= local_forks
        candidates[div_node] = {
            pred_div
            for pred_div in local_forks - invalid_forks
            if _is_strongly_connected_division(pred, pred_div, parent_ids, daughter_ids)
        }

    pairing = _bipartite_max_matching(list(candidates), candidates)
    scores = {div: int(div in pairing) for div in candidates}
    tp_forks = set(pairing.values())
    fp_forks = (considered | evaluable_forks | invalid_forks) - tp_forks
    return DivisionScores(scores=scores, tp_forks=tp_forks, fp_forks=fp_forks)


def division_counts(
    pred: TrackingGraph,
    gt: TrackingGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> DivisionCounts:
    """Compute division TP / FN / FP for a (pred, gt) pair."""
    result = score_divisions(pred, gt, scale=scale, max_distance=max_distance)
    tp = sum(result.scores.values())
    fn = len(result.scores) - tp
    return DivisionCounts(tp=tp, fn=fn, fp=len(result.fp_forks))
