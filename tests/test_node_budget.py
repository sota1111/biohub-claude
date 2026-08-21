"""Adjusted-Jaccard node-budget operating-point calibration (SOT-2912).

Pure-function unit tests — no competition data. Cover the node-budget penalty
algebra and the adjusted-objective operating-point selection / verdict logic.
"""

from __future__ import annotations

import math

import pytest

from biohub_tracking.eval.node_budget import (
    OperatingPoint,
    gt_node_count,
    node_budget_penalty,
    penalty_free_pred_nodes,
    select_adjusted_operating_point,
)
from biohub_tracking.graph import TrackingGraph


def test_penalty_sign_tracks_the_node_budget() -> None:
    # Over budget → penalty < 1 (matches the metric's 1 - 0.1*(110-100)/100).
    assert node_budget_penalty(110, 100) == pytest.approx(0.99)
    # At budget → no penalty.
    assert node_budget_penalty(100, 100) == pytest.approx(1.0)
    # Under budget → factor > 1 (a bonus; the metric is unclamped above 1).
    assert node_budget_penalty(50, 100) == pytest.approx(1.05)


def test_penalty_nan_without_budget() -> None:
    assert math.isnan(node_budget_penalty(100, float("nan")))
    assert math.isnan(node_budget_penalty(100, 0))
    assert math.isnan(node_budget_penalty(100, -5))


def test_penalty_free_pred_nodes_is_the_true_budget() -> None:
    assert penalty_free_pred_nodes(6362.0) == 6362.0


def test_gt_node_count_counts_gt_nodes() -> None:
    g = TrackingGraph.from_lists(
        {0: (0.0, 0.0, 0.0, 0.0), 1: (1.0, 0.0, 0.0, 0.0), 2: (2.0, 0.0, 0.0, 0.0)},
        [(0, 1), (1, 2)],
    )
    assert gt_node_count(g) == 3


def _pt(label, adj, raw, *, macro=None, lin=None, nra=True, nrr=True) -> OperatingPoint:
    return OperatingPoint(
        label=label,
        micro_adj=adj,
        micro_raw=raw,
        macro_adj=macro if macro is not None else adj,
        lineage_macro_adj=lin if lin is not None else adj,
        no_regression_adj=nra,
        no_regression_raw=nrr,
        total_pred_nodes=1000,
        total_true_nodes=900.0,
    )


CHAMP = _pt("min_track_length=4", 0.6649, 0.6142, macro=0.7180, lin=0.7216)


def test_rejected_when_no_adjusted_gain() -> None:
    # Higher mtl lifts RAW but regresses a family on ADJUSTED (the champion note's
    # mtl=5 case): raw-optimal diverges from adjusted-optimal, verdict rejected.
    cands = [
        _pt("min_track_length=5", 0.6600, 0.6692, macro=0.70, lin=0.70, nra=False),
        _pt("min_track_length=6", 0.6500, 0.6650, nra=False),
    ]
    sel = select_adjusted_operating_point(cands, CHAMP)
    assert sel.verdict == "rejected"
    # Adjusted objective is a different surface than the raw one.
    assert sel.raw_optimal_label == "min_track_length=5"
    assert sel.adj_optimal_label == "min_track_length=4"  # champion stays adj-best
    assert sel.raw_adj_divergent is True


def test_promoted_requires_gain_nonregression_and_mix_robustness() -> None:
    cands = [
        _pt("mad_k=3.5", 0.6700, 0.6200, macro=0.7250, lin=0.7300, nra=True, nrr=True),
    ]
    sel = select_adjusted_operating_point(cands, CHAMP)
    assert sel.verdict == "promoted"
    assert sel.best.label == "mad_k=3.5"
    assert sel.delta_micro_adj == pytest.approx(0.0051, abs=1e-4)


def test_inconclusive_when_gain_is_family_mix_fragile() -> None:
    # Micro-adj gain + adjusted 4/4, but the macro/lineage view does not rise
    # (family-mix noise) → not a safe promote.
    cands = [
        _pt("mad_k=3.5", 0.6700, 0.6200, macro=0.7100, lin=0.7100, nra=True, nrr=True),
    ]
    sel = select_adjusted_operating_point(cands, CHAMP)
    assert sel.verdict == "inconclusive"


def test_inconclusive_when_raw_guardrail_regresses() -> None:
    cands = [
        _pt("mad_k=3.5", 0.6700, 0.6200, macro=0.7250, lin=0.7300, nra=True, nrr=False),
    ]
    sel = select_adjusted_operating_point(cands, CHAMP)
    assert sel.verdict == "inconclusive"


def test_empty_candidates_raises() -> None:
    with pytest.raises(ValueError):
        select_adjusted_operating_point([], CHAMP)
