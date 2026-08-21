"""Adjusted-Jaccard node-budget operating-point calibration (SOT-2912).

The official royerlab metric is not the raw edge Jaccard but the **adjusted**
edge Jaccard, which scales the raw Jaccard by an explicit node-budget penalty

    J_adj = max(0, J · (1 − α · (N_pred − N_true) / N_true)),   α = 0.1

(see :mod:`biohub_tracking.eval.score`). The penalty is the piece no prior axis
optimised *as the objective*: earlier operating-point work (SOT-2789) swept the
**raw** metric and was declared exhausted there. This module optimises the
**adjusted** objective directly — it makes the node budget a first-class knob.

Two things fall out of the penalty algebra that make node-count a real lever:

* The factor is ``> 1`` when ``N_pred < N_true`` and ``< 1`` when
  ``N_pred > N_true``. So the adjusted metric *rewards* trimming predicted nodes
  down toward (or below) the true-node budget ``N_true`` — a trade the raw metric
  cannot see. Trimming almost always costs some raw TP edges, so there is a real
  interior optimum in node count, and it is generally **not** the raw-optimal
  operating point.
* ``N_true`` is the per-family true-node budget (the geff
  ``estimated_number_of_nodes`` metadata the metric divides by). This module also
  exposes a GT-density cross-check (:func:`gt_node_count`) so a caller can see how
  far the pred node count sits from the actual GT node count, not only the coarse
  metadata estimate.

The design split mirrors the rest of :mod:`biohub_tracking.eval`: everything here
is a **pure function** of already-computed numbers / graphs, so it unit-tests
without the (gitignored) competition ``data/``. The disk-touching sweep that
caches detections and re-links per operating point lives in
``experiments/sot2912/`` and calls :func:`select_adjusted_operating_point` to make
the promotion decision.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from ..graph import TrackingGraph
from .score import ADJUSTMENT_ALPHA


def gt_node_count(gt: TrackingGraph) -> int:
    """Number of ground-truth nodes — a GT-density view of the true-node budget.

    The adjusted metric divides by the coarse geff ``estimated_number_of_nodes``
    metadata; this is the *actual* GT node count, used as a cross-check that the
    metadata budget the operating point is calibrated against is not wildly off.
    """
    return gt.num_nodes


def node_budget_penalty(
    num_pred_nodes: int, n_true: float, alpha: float = ADJUSTMENT_ALPHA
) -> float:
    """The raw (unclamped) node-budget penalty factor ``1 − α·(N_pred−N_true)/N_true``.

    Returns NaN when ``n_true`` is missing/non-positive (the metric then drops the
    penalty and falls back to the raw Jaccard). This is the *unclamped* factor —
    the scored metric additionally clamps ``J_adj`` at 0 via ``max(0, …)`` — so a
    caller can read the sign/magnitude of the node-budget pressure directly:
    ``> 1`` means under budget (a bonus), ``< 1`` means over budget (a penalty).
    """
    if n_true is None or n_true <= 0 or math.isnan(n_true):
        return float("nan")
    return 1.0 - alpha * (num_pred_nodes - n_true) / n_true


def penalty_free_pred_nodes(n_true: float) -> float:
    """The largest predicted-node count that incurs **no** node-budget penalty.

    Equals ``N_true``: at or below the true-node budget the penalty factor is
    ``>= 1`` (no penalty; a small bonus below), above it the factor drops. This is
    the node-count target the adjusted objective calibrates the operating point
    toward — the raw metric has no such target.
    """
    return float(n_true)


class OperatingPoint(NamedTuple):
    """One calibrated operating point on the adjusted objective.

    ``label`` identifies the swept knob value (e.g. ``"min_track_length=5"``).
    ``micro_adj`` / ``micro_raw`` are the CV micro-averaged adjusted / raw edge
    Jaccards; ``no_regression_adj`` / ``no_regression_raw`` are the per-dataset
    4/4 non-regression guards vs the incumbent champion (adjusted primary, raw
    guardrail — SOT-2903). ``total_pred_nodes`` / ``total_true_nodes`` expose the
    realised node budget so the penalty pressure is auditable.
    """

    label: str
    micro_adj: float
    micro_raw: float
    macro_adj: float
    lineage_macro_adj: float
    no_regression_adj: bool
    no_regression_raw: bool
    total_pred_nodes: int
    total_true_nodes: float


class OperatingPointSelection(NamedTuple):
    """Result of an adjusted-objective operating-point search.

    ``verdict`` is one of ``promoted`` / ``rejected`` / ``inconclusive`` under the
    same discipline the biohub linking A/Bs use (a promote needs a strict
    micro-adj gain AND 4/4 adjusted non-regression AND family-mix robustness AND a
    non-regressing raw guardrail). ``raw_adj_divergent`` records the axis's core
    finding: whether the raw-optimal operating point differs from the
    adjusted-optimal one — the evidence that this is an adjusted-objective axis and
    not a rerun of the raw sweep.
    """

    best: OperatingPoint
    champion: OperatingPoint
    raw_optimal_label: str
    adj_optimal_label: str
    raw_adj_divergent: bool
    delta_micro_adj: float
    delta_micro_raw: float
    verdict: str


def select_adjusted_operating_point(
    points: list[OperatingPoint],
    champion: OperatingPoint,
    *,
    tol: float = 1e-4,
) -> OperatingPointSelection:
    """Pick the adjusted-objective-optimal operating point and rule on promotion.

    ``points`` are the swept candidate operating points (excluding the champion);
    ``champion`` is the incumbent's operating point (its ``no_regression_*`` fields
    are ignored — it is the reference). Selection prefers the adjusted
    non-regressing candidate with the highest ``micro_adj``; if none is
    non-regressing it falls back to the overall highest ``micro_adj`` so the
    divergence diagnostics are still reported.

    Promotion discipline (identical to the SOT-2910/2903 linking A/Bs):

    * ``promoted``  — strict micro-adj gain over the champion beyond ``tol``, AND
      4/4 adjusted non-regression, AND family-mix robustness (macro *and*
      lineage-macro both up), AND the raw guardrail does not regress any family.
    * ``rejected``  — no micro-adj gain, or the best point regresses a family on
      the adjusted metric.
    * ``inconclusive`` — a micro-adj gain that is family-mix fragile (macro or
      lineage-macro not up, or the raw guardrail regresses).
    """
    if not points:
        raise ValueError("no candidate operating points to select from")

    # Adjusted-optimal: prefer the adjusted-non-regressing pool, else all points.
    non_reg = [p for p in points if p.no_regression_adj]
    pool = non_reg or points
    best = max(pool, key=lambda p: p.micro_adj)

    # Raw-optimal over the SAME candidate pool (champion included, so a champion
    # that is already raw-best is detected): this is the point the *raw* sweep
    # (SOT-2789) would have chosen. Divergence => the adjusted objective is a
    # genuinely different selection surface than the raw one.
    raw_pool = [champion, *points]
    raw_optimal = max(raw_pool, key=lambda p: p.micro_raw)
    adj_pool = [champion, *points]
    adj_optimal = max(adj_pool, key=lambda p: p.micro_adj)
    raw_adj_divergent = raw_optimal.label != adj_optimal.label

    delta_micro_adj = best.micro_adj - champion.micro_adj
    delta_micro_raw = best.micro_raw - champion.micro_raw

    macro_up = best.macro_adj > champion.macro_adj
    lineage_up = best.lineage_macro_adj > champion.lineage_macro_adj

    if delta_micro_adj <= tol or not best.no_regression_adj:
        verdict = "rejected"
    elif macro_up and lineage_up and best.no_regression_raw:
        verdict = "promoted"
    else:
        verdict = "inconclusive"

    return OperatingPointSelection(
        best=best,
        champion=champion,
        raw_optimal_label=raw_optimal.label,
        adj_optimal_label=adj_optimal.label,
        raw_adj_divergent=raw_adj_divergent,
        delta_micro_adj=round(delta_micro_adj, 6),
        delta_micro_raw=round(delta_micro_raw, 6),
        verdict=verdict,
    )
