"""Golden round-trip checks for the deterministic synthetic division fixture."""

from __future__ import annotations

from biohub_tracking.eval import division_counts, inject_synthetic_division
from biohub_tracking.graph import TrackingGraph


def _linear_graph() -> TrackingGraph:
    return TrackingGraph.from_lists(
        {node_id: (node_id, 0, 0, node_id) for node_id in range(6)},
        [(node_id, node_id + 1) for node_id in range(5)],
    )


def test_synthetic_division_is_deterministic_and_inside_matching_window() -> None:
    source = _linear_graph()
    first = inject_synthetic_division(source)
    second = inject_synthetic_division(source)

    assert first == second
    assert source.dividing_nodes() == []
    assert first.graph.dividing_nodes() == [first.parent_id]
    assert division_counts(first.graph, first.graph) == (1, 0, 0)


def test_synthetic_division_round_trip_tp_fn_fp_golden() -> None:
    fixture = inject_synthetic_division(_linear_graph())
    gt = fixture.graph

    # Original linear graph misses the injected event.
    assert division_counts(_linear_graph(), gt) == (0, 1, 0)

    # An extra evaluable fork produces one true and one false positive event.
    pred = TrackingGraph.from_lists(dict(gt.coords), list(gt.edges))
    extra_id = max(pred.node_ids()) + 1
    t, z, y, x = pred.coords[4]
    pred.add_node(extra_id, t, z, y + 3.0, x)
    pred.add_edge(3, extra_id)
    assert division_counts(pred, gt) == (1, 0, 1)
