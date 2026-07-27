from __future__ import annotations

from biohub_tracking.graph import TrackingGraph
from scripts.detection_sanity import summarize_graph


def test_summarize_graph_records_required_sanity_metrics() -> None:
    graph = TrackingGraph.from_lists(
        {
            0: (0, 0, 0, 0),
            1: (1, 0, 0, 0),
            2: (0, 0, 5, 0),
            3: (1, 0, 5, 0),
        },
        [(0, 1), (2, 3)],
    )
    result = summarize_graph(graph)

    assert result["node_count"] == 4
    assert result["track_lengths"] == {"count": 2, "min": 2, "median": 2.0, "max": 2}
    assert result["detections_per_frame"] == {"0": 2, "1": 2}
    assert result["sanity"]["pass"] is True
