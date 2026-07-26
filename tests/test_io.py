"""Submission-CSV round-trip and geff loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from biohub_tracking.graph import TrackingGraph
from biohub_tracking.io import (
    CSV_COLUMNS,
    graph_to_rows,
    load_geff,
    load_submission_csv,
    write_submission_csv,
)

_REAL_GEFF = Path(__file__).parent.parent / "data" / "train" / "44b6_0113de3b.geff"


def _sample_graph() -> TrackingGraph:
    return TrackingGraph.from_lists(
        nodes={
            1: (0, 32.0, 128.0, 128.0),
            2: (1, 33.0, 130.0, 131.0),
            3: (2, 34.0, 132.0, 129.0),
        },
        edges=[(1, 2), (2, 3)],
    )


def test_csv_round_trip_is_exact(tmp_path: Path) -> None:
    g = _sample_graph()
    csv_path = tmp_path / "sub.csv"
    write_submission_csv({"ds1": g}, csv_path)

    back = load_submission_csv(csv_path)["ds1"]
    assert back.num_nodes == g.num_nodes
    assert back.num_edges == g.num_edges
    assert back.coords == g.coords
    assert set(back.edges) == set(g.edges)


def test_csv_header_and_row_shape(tmp_path: Path) -> None:
    csv_path = tmp_path / "sub.csv"
    write_submission_csv({"ds1": _sample_graph()}, csv_path)
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == ",".join(CSV_COLUMNS)
    # 3 nodes + 2 edges = 5 data rows, ids 0..4.
    assert len(lines) == 1 + 5
    assert lines[1].startswith("0,ds1,node,")


def test_graph_to_rows_marks_unused_fields_minus_one() -> None:
    rows = graph_to_rows(_sample_graph(), "ds1")
    node_rows = [r for r in rows if r["row_type"] == "node"]
    edge_rows = [r for r in rows if r["row_type"] == "edge"]
    assert all(r["source_id"] == -1 and r["target_id"] == -1 for r in node_rows)
    assert all(
        r["node_id"] == -1 and r["t"] == -1 and r["z"] == -1 and r["y"] == -1 and r["x"] == -1
        for r in edge_rows
    )
    # Nodes precede edges in the flattened output.
    assert [r["row_type"] for r in rows] == ["node"] * 3 + ["edge"] * 2


def test_multiple_datasets_are_separated(tmp_path: Path) -> None:
    g1 = _sample_graph()
    g2 = TrackingGraph.from_lists({5: (0, 0.0, 0.0, 0.0)}, [])
    csv_path = tmp_path / "sub.csv"
    write_submission_csv({"a": g1, "b": g2}, csv_path)
    graphs = load_submission_csv(csv_path)
    assert set(graphs) == {"a", "b"}
    assert graphs["b"].num_nodes == 1 and graphs["b"].num_edges == 0


@pytest.mark.skipif(not _REAL_GEFF.exists(), reason="real competition geff not present")
def test_load_real_geff() -> None:
    g = load_geff(_REAL_GEFF)
    assert g.num_nodes == 52
    assert g.num_edges == 50
    # Every edge is a single forward time step in this sparse annotation.
    assert all(g.t(t) == g.t(s) + 1 for s, t in g.edges)
