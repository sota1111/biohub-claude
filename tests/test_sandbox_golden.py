"""Golden edge + division counts on small hand-crafted graphs.

These cases and their frozen expected TP/FP/FN are ported from the official
reference test suite (``royerlab/kaggle-cell-tracking-competition``, BSD-3-Clause,
``tests/test_division_sandbox_examples.py``). Reproducing every one of them gives
strong confidence this lightweight reimplementation matches the real Kaggle
metric.

Graphs are y-only (z == x == 0) and matched isotropically (scale=None), so the
centroid distance reduces to ``|Δy|``.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from biohub_tracking.eval import division_counts, edge_counts, evaluate, score_divisions
from biohub_tracking.graph import TrackingGraph
from biohub_tracking.matching import match_nodes


class GraphSpec(NamedTuple):
    nodes: dict[str, tuple[int, float]]  # name -> (t, y)
    edges: list[tuple[str, str]]


class Expected(NamedTuple):
    edge_tp: int
    edge_fp: int
    edge_fn: int
    div_tp: int
    div_fp: int
    div_fn: int


class Case(NamedTuple):
    gt: GraphSpec
    pred: GraphSpec
    expected: Expected
    max_distance: float = 1.0


_GT_DIVISION = GraphSpec(
    nodes={
        "P": (0, 0.0), "D": (1, 0.0),
        "C1": (2, 5.0), "C2": (2, -5.0),
        "G1": (3, 5.0), "G2": (3, -5.0),
    },
    edges=[("P", "D"), ("D", "C1"), ("D", "C2"), ("C1", "G1"), ("C2", "G2")],
)


CASES: dict[str, Case] = {
    "perfect_division": Case(
        gt=_GT_DIVISION, pred=_GT_DIVISION,
        expected=Expected(5, 0, 0, 1, 0, 0),
    ),
    "missed_division": Case(
        gt=_GT_DIVISION,
        pred=GraphSpec(
            nodes={"P": (0, 0.0), "D": (1, 0.0), "C1": (2, 5.0), "G1": (3, 5.0)},
            edges=[("P", "D"), ("D", "C1"), ("C1", "G1")],
        ),
        expected=Expected(3, 0, 2, 0, 0, 1),
    ),
    "delayed_local_division": Case(
        gt=_GT_DIVISION,
        pred=GraphSpec(
            nodes={"P": (0, 0.0), "D": (1, 0.0), "M": (2, 0.0), "G1": (3, 5.0), "G2": (3, -5.0)},
            edges=[("P", "D"), ("D", "M"), ("M", "G1"), ("M", "G2")],
        ),
        expected=Expected(1, 3, 4, 1, 0, 0),
    ),
    "dummy_branch_same_lineage": Case(
        gt=_GT_DIVISION,
        pred=GraphSpec(
            nodes={"P": (0, 0.0), "D": (1, 0.0), "C1": (2, 5.0), "X": (2, 50.0), "G2": (3, -5.0)},
            edges=[("P", "D"), ("D", "C1"), ("D", "X"), ("C1", "G2")],
        ),
        expected=Expected(2, 2, 3, 0, 1, 1),
    ),
    "spurious_linear_division": Case(
        gt=GraphSpec(
            nodes={"A": (0, 20.0), "B": (1, 20.0), "C": (2, 20.0)},
            edges=[("A", "B"), ("B", "C")],
        ),
        pred=GraphSpec(
            nodes={"A": (0, 20.0), "B": (1, 20.0), "C1": (2, 20.0), "C2": (2, 25.0)},
            edges=[("A", "B"), ("B", "C1"), ("B", "C2")],
        ),
        expected=Expected(2, 1, 0, 0, 1, 0),
    ),
    "cross_component_children": Case(
        gt=GraphSpec(
            nodes={"A0": (0, 0.0), "A1": (1, 0.0), "B0": (0, 20.0), "B1": (1, 20.0)},
            edges=[("A0", "A1"), ("B0", "B1")],
        ),
        pred=GraphSpec(
            nodes={"P": (0, 0.0), "C1": (1, 0.0), "C2": (1, 20.0)},
            edges=[("P", "C1"), ("P", "C2")],
        ),
        expected=Expected(1, 1, 1, 0, 1, 0),
    ),
    "cross_component_grandchild_fallback": Case(
        gt=GraphSpec(
            nodes={"A1": (1, 0.0), "A2": (2, 0.0), "B1": (1, 20.0), "B2": (2, 20.0)},
            edges=[("A1", "A2"), ("B1", "B2")],
        ),
        pred=GraphSpec(
            nodes={"F": (0, 10.0), "A": (1, 0.0), "U": (1, 40.0), "B": (2, 20.0)},
            edges=[("F", "A"), ("F", "U"), ("U", "B")],
        ),
        expected=Expected(0, 1, 2, 0, 1, 0),
    ),
    "disconnected_daughter": Case(
        gt=_GT_DIVISION,
        pred=GraphSpec(
            nodes=_GT_DIVISION.nodes,
            edges=[("P", "D"), ("D", "C1"), ("C1", "G1"), ("C2", "G2")],
        ),
        expected=Expected(4, 0, 1, 0, 0, 1),
    ),
    "hack2": Case(
        gt=GraphSpec(
            nodes={
                "61": (1, 10.0), "62": (2, 10.0), "63": (3, 0.0), "64": (3, 20.0),
                "66": (4, 20.0), "98": (4, 0.0), "99": (1, -40.0), "100": (2, -40.0),
                "101": (3, -25.0), "102": (4, -25.0), "103": (3, -45.0), "104": (4, -45.0),
            },
            edges=[
                ("61", "62"), ("62", "63"), ("62", "64"), ("64", "66"), ("63", "98"),
                ("99", "100"), ("100", "101"), ("101", "102"), ("100", "103"), ("103", "104"),
            ],
        ),
        pred=GraphSpec(
            nodes={
                "105": (0, 15.0), "106": (2, 15.0), "107": (3, 30.0), "108": (3, 5.0),
                "109": (4, 5.0), "110": (4, 30.0), "114": (4, -20.0), "115": (3, -50.0),
                "116": (4, -50.0), "121": (5, -50.0), "122": (1, 55.0), "124": (2, 55.0),
                "130": (3, 55.0), "132": (1, 15.0), "153": (2, -35.0),
            },
            edges=[
                ("106", "107"), ("108", "109"), ("107", "110"), ("115", "116"),
                ("116", "121"), ("122", "124"), ("124", "108"), ("124", "130"),
                ("130", "114"), ("105", "132"), ("132", "106"), ("105", "122"),
                ("122", "153"), ("106", "115"),
            ],
        ),
        expected=Expected(5, 4, 5, 0, 4, 2),
        max_distance=10.0,
    ),
}


def _build(spec: GraphSpec) -> tuple[TrackingGraph, dict[str, int]]:
    ids = {name: i for i, name in enumerate(spec.nodes)}
    nodes = {ids[name]: (float(t), 0.0, float(y), 0.0) for name, (t, y) in spec.nodes.items()}
    edges = [(ids[s], ids[t]) for s, t in spec.edges]
    return TrackingGraph.from_lists(nodes, edges), ids


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_counts(name: str) -> None:
    case = CASES[name]
    pred, _ = _build(case.pred)
    gt, _ = _build(case.gt)

    matching = match_nodes(pred, gt, scale=None, max_distance=case.max_distance)
    ec = edge_counts(pred, gt, matching)
    dc = division_counts(pred, gt, scale=None, max_distance=case.max_distance)

    got = Expected(ec.tp, ec.fp, ec.fn, dc.tp, dc.fp, dc.fn)
    assert got == case.expected, f"{name}: {got} != {case.expected}"


@pytest.mark.parametrize("name", sorted(CASES))
def test_evaluate_agrees(name: str) -> None:
    case = CASES[name]
    pred, _ = _build(case.pred)
    gt, _ = _build(case.gt)
    r = evaluate(pred, gt, scale=None, max_distance=case.max_distance)
    assert (r.edge_tp, r.edge_fp, r.edge_fn) == (
        case.expected.edge_tp, case.expected.edge_fp, case.expected.edge_fn,
    )
    assert (r.division_tp, r.division_fp, r.division_fn) == (
        case.expected.div_tp, case.expected.div_fp, case.expected.div_fn,
    )


def test_hack2_fp_fork_identities() -> None:
    """Freeze the exact exploit topology's four false-positive forks."""
    case = CASES["hack2"]
    pred, pred_ids = _build(case.pred)
    gt, _ = _build(case.gt)
    result = score_divisions(pred, gt, max_distance=case.max_distance)
    fp_names = {name for name, nid in pred_ids.items() if nid in result.fp_forks}
    assert fp_names == {"105", "106", "122", "124"}
