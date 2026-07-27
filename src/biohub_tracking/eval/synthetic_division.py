"""Deterministically add one measurable division to a linear tracking graph."""

from __future__ import annotations

from dataclasses import dataclass

from ..graph import TrackingGraph


@dataclass(frozen=True)
class SyntheticDivision:
    """The generated graph and identifiers of the injected event."""

    graph: TrackingGraph
    parent_id: int
    existing_daughter_id: int
    synthetic_daughter_id: int
    synthetic_granddaughter_id: int | None


def inject_synthetic_division(
    source: TrackingGraph,
    *,
    scale: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    max_distance: float = 7.0,
    offset_um: float = 4.0,
) -> SyntheticDivision:
    """Return a copy of *source* with one ``parent -> 2 daughters`` event.

    The earliest fully-windowed linear edge is selected, making the result
    independent of dictionary insertion order.  The synthetic daughter is
    offset along x by ``offset_um`` and therefore remains inside the metric's
    physical matching window.  A matching granddaughter is added when the
    existing daughter has a successor, so the complete local division window
    can be exercised.
    """
    if not 0.0 < offset_um <= max_distance:
        raise ValueError("offset_um must be positive and no greater than max_distance")
    if len(scale) != 3 or any(value <= 0 for value in scale):
        raise ValueError("scale must contain three positive values")

    candidates = [
        node_id
        for node_id in source.node_ids()
        if source.in_degree(node_id) == 1
        and source.out_degree(node_id) == 1
        and source.in_degree(source.successors(node_id)[0]) == 1
        and source.out_degree(source.successors(node_id)[0]) == 1
    ]
    if not candidates:
        raise ValueError("source graph has no fully-windowed linear edge")
    parent_id = min(candidates, key=lambda node_id: (source.t(node_id), node_id))
    existing_daughter_id = source.successors(parent_id)[0]
    existing_granddaughter_id = source.successors(existing_daughter_id)[0]

    graph = TrackingGraph.from_lists(dict(source.coords), list(source.edges))
    next_id = max(graph.node_ids(), default=-1) + 1
    synthetic_daughter_id = next_id
    synthetic_granddaughter_id = next_id + 1
    offset_voxels = offset_um / scale[2]

    t, z, y, x = graph.coords[existing_daughter_id]
    graph.add_node(synthetic_daughter_id, t, z, y, x + offset_voxels)
    graph.add_edge(parent_id, synthetic_daughter_id)

    gt, gz, gy, gx = graph.coords[existing_granddaughter_id]
    graph.add_node(synthetic_granddaughter_id, gt, gz, gy, gx + offset_voxels)
    graph.add_edge(synthetic_daughter_id, synthetic_granddaughter_id)

    return SyntheticDivision(
        graph=graph,
        parent_id=parent_id,
        existing_daughter_id=existing_daughter_id,
        synthetic_daughter_id=synthetic_daughter_id,
        synthetic_granddaughter_id=synthetic_granddaughter_id,
    )
