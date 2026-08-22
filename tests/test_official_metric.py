"""Official-scorer bridge conformance (SOT-2995).

Two independent guarantees:

* ``graph_to_geff`` round-trips through :func:`biohub_tracking.io.load_geff`
  losslessly — a LIGHT test (numpy + zarr only) that always runs.
* The genuine royerlab scorer (``tracksdata`` + ``tracking_cellmot``), invoked
  through :mod:`biohub_tracking.eval.official`, reproduces every frozen golden
  TP/FP/FN AND agrees count-for-count with the clean-room scorer on the same
  graphs. These are ``importorskip``-guarded, so CI without the heavy dev-only
  stack skips them instead of failing; run them in the official venv to prove
  oracle fidelity.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from biohub_tracking.eval.official import graph_to_geff, official_available
from biohub_tracking.graph import TrackingGraph
from biohub_tracking.io import (
    geff_estimated_num_nodes,
    geff_scale,
    load_geff,
)

from test_sandbox_golden import CASES, _build


def test_official_module_imports_without_heavy_deps() -> None:
    """Importing the bridge (and calling the probe) must never require tracksdata."""
    # official_available() is a pure probe; it returns a bool either way.
    assert official_available() in (True, False)


def test_graph_to_geff_round_trips() -> None:
    g = TrackingGraph.from_lists(
        {
            0: (0.0, 1.0, 5.0, 2.0),
            1: (1.0, 1.0, 5.0, 2.0),
            2: (2.0, 0.0, -5.0, 3.0),
        },
        [(0, 1), (1, 2)],
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sample.geff"
        graph_to_geff(
            g, path, scale=(1.625, 0.40625, 0.40625), estimated_number_of_nodes=99.0
        )
        loaded = load_geff(path)
        assert loaded.num_nodes == g.num_nodes
        assert loaded.num_edges == g.num_edges
        assert sorted(loaded.edges) == sorted(g.edges)
        # Coordinates are stored int64 in geff; the sample is integer-valued.
        assert loaded.coords == g.coords
        assert geff_scale(path) == (1.625, 0.40625, 0.40625)
        assert geff_estimated_num_nodes(path) == 99.0


@pytest.mark.parametrize("name", sorted(CASES))
def test_official_reproduces_golden_counts(name: str) -> None:
    """The GENUINE royerlab scorer reproduces every frozen golden TP/FP/FN."""
    pytest.importorskip("tracksdata")
    pytest.importorskip("tracking_cellmot")
    from biohub_tracking.eval.official import official_evaluate

    case = CASES[name]
    pred, _ = _build(case.pred)
    gt, _ = _build(case.gt)
    r = official_evaluate(pred, gt, scale=None, max_distance=case.max_distance)
    assert (r.edge_tp, r.edge_fp, r.edge_fn) == (
        case.expected.edge_tp,
        case.expected.edge_fp,
        case.expected.edge_fn,
    )
    assert (r.division_tp, r.division_fp, r.division_fn) == (
        case.expected.div_tp,
        case.expected.div_fp,
        case.expected.div_fn,
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_official_matches_clean_room(name: str) -> None:
    """Official and clean-room scorers agree count-for-count (zero divergence)."""
    pytest.importorskip("tracksdata")
    pytest.importorskip("tracking_cellmot")
    from biohub_tracking.eval.official import divergence_row

    case = CASES[name]
    pred, _ = _build(case.pred)
    gt, _ = _build(case.gt)
    row = divergence_row(name, pred, gt, scale=None, max_distance=case.max_distance)
    assert row.counts_match, (
        f"{name}: clean={tuple(row.clean)} official={tuple(row.official)}"
    )
