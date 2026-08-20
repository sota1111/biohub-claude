"""Single leak-free CV harness — the one primary-KPI evaluator (SOT-2761).

Every earlier experiment (``scripts/reanchor_oracle.py``,
``experiments/sot2369/confirm_shorttrack.py``, ``experiments/sot23xx/screen_*.py``)
re-declared the same four-family holdout list and re-implemented the same
micro-averaging by hand. That duplication is exactly how an oracle silently
drifts between screens. This module is the **single source of truth** for

* the **leak-free CV holdout** definition (:data:`CV_HOLDOUT`), and
* the aggregation that turns per-family counts into the reported CV score
  (:func:`aggregate` / :func:`evaluate_cv`),

so every later child Issue screens and confirms against *one* CV, returning the
**micro-adjusted edge Jaccard + per-dataset breakdown + division Jaccard**.

Leak-free design (see ``docs/ai/sot-2761-leak-free-cv.md`` for the full write-up):

* **Entity holdout.** The scored unit is a whole *embryo video* (a ``.geff``
  lineage), never an individual frame or node, so no cell/track straddles the
  train↔score boundary. The four families are the exact Kaggle test set —
  two ``44b6`` embryo videos + two ``6bba`` embryo videos — and the train split
  ships GT ``.geff`` for all four, so the CV target IS the LB target.
* **Temporal holdout.** The pipeline is deterministic and *causal*: detection is
  per-timepoint and linking is forward-only (frame ``t`` → ``t+1``), so no
  future frame informs an earlier prediction and no parameter is fit on future
  data. There is nothing learned across time to leak.
* **Selection discipline.** With only four videos, selecting on the micro alone
  overfits the holdout; callers must gate on *per-dataset no-regression*
  (:meth:`CvResult.no_regression_vs`) as well as the micro, exactly as the
  champion promotions did.

Data-free by construction: the aggregation (:func:`aggregate`) and per-family
scoring (:func:`score_family`) are pure functions of already-computed
:class:`TrackingGraph` objects, so they unit-test without the (gitignored)
competition data. Only :func:`evaluate_cv` touches disk, and only when the real
``data/`` volumes are present.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple

from ..graph import TrackingGraph
from ..matching import DEFAULT_MAX_DISTANCE
from .score import (
    SCORE_DIVISION_WEIGHT,
    _jaccard,
    adjusted_edge_jaccard,
    evaluate,
)


@dataclass(frozen=True)
class HoldoutFamily:
    """One leak-free CV holdout entity (a whole embryo video).

    ``lineage`` is the embryo-lineage prefix (``44b6`` / ``6bba``); it is the
    entity key the holdout is split on — two distinct embryo lineages, so the
    per-lineage breakdown shows whether a change generalises across embryos
    rather than overfitting the one lineage it was tuned on.
    """

    name: str
    lineage: str
    image: str  # test *.zarr volume (the scored input)
    geff: str  # train *.geff ground truth (labels; scoring only, never fit)


# The four families that make up the real Kaggle test set. For each we hold the
# test *.zarr image locally and the train *.geff ground truth. Entity split: two
# 44b6 embryo videos + two 6bba embryo videos, disjoint lineages. This is the
# exact scored set, so the micro here is the faithful LB oracle (SOT-2305).
CV_HOLDOUT: tuple[HoldoutFamily, ...] = (
    HoldoutFamily("44b6_0113de3b", "44b6", "data/test/44b6_0113de3b.zarr", "data/train/44b6_0113de3b.geff"),
    HoldoutFamily("44b6_0b24845f", "44b6", "data/test/44b6_0b24845f.zarr", "data/train/44b6_0b24845f.geff"),
    HoldoutFamily("6bba_05b6850b", "6bba", "data/test/6bba_05b6850b.zarr", "data/train/6bba_05b6850b.geff"),
    HoldoutFamily("6bba_05db0fb1", "6bba", "data/test/6bba_05db0fb1.zarr", "data/train/6bba_05db0fb1.geff"),
)

# Registry champion (detect-link-dog-v4-shorttrack) reference CV, so the harness
# self-checks that it still reproduces the promoted number deterministically.
CHAMPION_REFERENCE_MICRO_ADJ: float = 0.6649
PUBLIC_LB_BEST: float = 0.624  # champion public best, for CV↔public order check.


class FamilyResult(NamedTuple):
    """Per-dataset CV counts + derived Jaccards for one holdout family."""

    name: str
    lineage: str
    edge_tp: int
    edge_fp: int
    edge_fn: int
    edge_jaccard: float
    adj_edge_jaccard: float
    division_tp: int
    division_fp: int
    division_fn: int
    num_pred_nodes: int
    n_true: float
    weight: int  # tp + fp + fn — the micro-average sample weight


class CvResult(NamedTuple):
    """Aggregated CV: micro-adjusted edge Jaccard, division Jaccard, breakdowns."""

    micro_adj_edge_jaccard: float
    micro_edge_jaccard: float
    division_jaccard: float
    score: float
    per_dataset: tuple[FamilyResult, ...]
    by_lineage: dict[str, float]

    def no_regression_vs(self, incumbent: dict[str, float]) -> bool:
        """True iff every family's adjusted edge Jaccard is >= *incumbent*'s.

        The mandatory selection gate: with only four videos, a micro gain that
        hides a per-family regression is exactly the overfit the checklist warns
        against, so a promotion must not regress any single lineage/family.
        """
        return all(
            fr.adj_edge_jaccard >= incumbent.get(fr.name, float("-inf"))
            for fr in self.per_dataset
        )


def score_family(
    family: HoldoutFamily,
    pred: TrackingGraph,
    gt: TrackingGraph,
    n_true: float,
    scale: tuple[float, ...] | None = None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> FamilyResult:
    """Score one already-predicted family against its GT (pure; no disk I/O)."""
    r = evaluate(pred, gt, scale=scale, max_distance=max_distance)
    j = _jaccard(r.edge_tp, r.edge_fp, r.edge_fn)
    adj = adjusted_edge_jaccard(j, r.num_pred_nodes, n_true)
    if math.isnan(adj):  # no node estimate → fall back to the raw Jaccard
        adj = j
    return FamilyResult(
        name=family.name,
        lineage=family.lineage,
        edge_tp=r.edge_tp,
        edge_fp=r.edge_fp,
        edge_fn=r.edge_fn,
        edge_jaccard=j,
        adj_edge_jaccard=adj,
        division_tp=r.division_tp,
        division_fp=r.division_fp,
        division_fn=r.division_fn,
        num_pred_nodes=r.num_pred_nodes,
        n_true=n_true,
        weight=r.edge_tp + r.edge_fp + r.edge_fn,
    )


def aggregate(rows: list[FamilyResult] | tuple[FamilyResult, ...]) -> CvResult:
    """Micro-average per-family rows into the reported CV (pure function).

    Aggregation matches :mod:`biohub_tracking.eval.score` exactly: edge/division
    Jaccards are micro-averaged (sum TP/FP/FN then divide); the adjusted edge
    Jaccard is the per-family adjusted Jaccard weight-averaged by sample size
    ``w = tp + fp + fn``. This is the identical arithmetic the champion
    promotions used, so the champion still reproduces its registry 0.6649.
    """
    rows = tuple(rows)
    edge_tp = sum(r.edge_tp for r in rows)
    edge_fp = sum(r.edge_fp for r in rows)
    edge_fn = sum(r.edge_fn for r in rows)
    div_tp = sum(r.division_tp for r in rows)
    div_fp = sum(r.division_fp for r in rows)
    div_fn = sum(r.division_fn for r in rows)

    micro_edge_jaccard = _jaccard(edge_tp, edge_fp, edge_fn)

    def _weighted_adj(subset: tuple[FamilyResult, ...]) -> float:
        wsum = sum(r.weight * r.adj_edge_jaccard for r in subset if r.weight > 0)
        wtot = sum(r.weight for r in subset if r.weight > 0)
        return wsum / wtot if wtot > 0 else float("nan")

    micro_adj = _weighted_adj(rows)

    by_lineage = {
        lin: _weighted_adj(tuple(r for r in rows if r.lineage == lin))
        for lin in sorted({r.lineage for r in rows})
    }

    has_div = (div_tp + div_fp + div_fn) > 0
    if has_div:
        division_jaccard = _jaccard(div_tp, div_fp, div_fn)
        score = micro_adj + SCORE_DIVISION_WEIGHT * division_jaccard
    else:
        division_jaccard = float("nan")
        score = micro_adj

    return CvResult(
        micro_adj_edge_jaccard=micro_adj,
        micro_edge_jaccard=micro_edge_jaccard,
        division_jaccard=division_jaccard,
        score=score,
        per_dataset=rows,
        by_lineage=by_lineage,
    )


# Signature of the pluggable predictor: (image_path, scale, detect, link) → graph.
Predictor = Callable[..., TrackingGraph]


def evaluate_cv(
    config: dict | None = None,
    *,
    repo_root: Path | str | None = None,
    families: tuple[HoldoutFamily, ...] = CV_HOLDOUT,
    predictor: Predictor | None = None,
) -> CvResult:
    """Run the champion (or *config*) over the CV holdout and aggregate.

    This is the single entry point later child Issues call to get the CV score.
    It loads each family's GT/scale/node-estimate, runs the detection+linking
    pipeline over the family's test volume, scores it, and micro-averages.

    Parameters
    ----------
    config
        A champion config dict (as in ``champion/config.json``). ``None`` loads
        the reigning champion via :func:`biohub_tracking.champion.load_champion_config`.
    repo_root
        Repo root the family relative paths resolve against. Defaults to the repo
        that contains this module.
    predictor
        Optional injection for testing — ``(image, scale, detect, link) → graph``.
        Defaults to the real :func:`biohub_tracking.pipeline.run_pipeline`.

    Requires the competition ``data/`` volumes on disk; it is not exercised in CI
    (the data is gitignored). The pure aggregation is covered by unit tests.
    """
    # Local imports: these pull in champion/pipeline/io (and zarr) which are not
    # needed for the pure aggregation path the unit tests exercise.
    from ..champion import champion_params, load_champion_config
    from ..io import geff_estimated_num_nodes, geff_scale, load_geff

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    repo_root = Path(repo_root)

    if config is None:
        config = load_champion_config()
    detect, link, _cfg_scale = champion_params(config)

    if predictor is None:
        from ..pipeline import run_pipeline

        def predictor(image, scale, detect_params, link_params):  # noqa: ANN001
            return run_pipeline(
                image, scale=scale, detect_params=detect_params, link_params=link_params
            )

    rows: list[FamilyResult] = []
    for fam in families:
        geff = repo_root / fam.geff
        image = repo_root / fam.image
        gt = load_geff(geff)
        scale = geff_scale(geff)
        n_true = geff_estimated_num_nodes(geff)
        pred = predictor(image, scale, detect, link)
        rows.append(score_family(fam, pred, gt, n_true, scale=scale))
    return aggregate(rows)


def cv_result_to_dict(result: CvResult) -> dict:
    """JSON-serialisable view of a :class:`CvResult` (rounded for reports)."""
    return {
        "micro_adj_edge_jaccard": round(result.micro_adj_edge_jaccard, 4),
        "micro_edge_jaccard": round(result.micro_edge_jaccard, 4),
        "division_jaccard": (
            None
            if math.isnan(result.division_jaccard)
            else round(result.division_jaccard, 4)
        ),
        "score": round(result.score, 4),
        "by_lineage": {k: round(v, 4) for k, v in result.by_lineage.items()},
        "per_dataset": [
            {
                "name": r.name,
                "lineage": r.lineage,
                "edge_tp": r.edge_tp,
                "edge_fp": r.edge_fp,
                "edge_fn": r.edge_fn,
                "edge_jaccard": round(r.edge_jaccard, 4),
                "adjusted_edge_jaccard": round(r.adj_edge_jaccard, 4),
                "division_tp": r.division_tp,
                "division_fp": r.division_fp,
                "division_fn": r.division_fn,
                "pred_nodes": r.num_pred_nodes,
                "n_true": (None if math.isnan(r.n_true) else r.n_true),
                "weight": r.weight,
            }
            for r in result.per_dataset
        ],
    }


def _main(argv: list[str] | None = None) -> int:
    """CLI: run the champion CV and print (optionally write) the JSON result."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Run the leak-free CV harness (SOT-2761).")
    ap.add_argument("--out", type=str, default=None, help="Write the JSON result here.")
    ap.add_argument(
        "--check-champion",
        action="store_true",
        help="Assert the champion reproduces the registry reference micro-adj.",
    )
    ap.add_argument(
        "--tol", type=float, default=1e-4, help="Tolerance for --check-champion."
    )
    args = ap.parse_args(argv)

    result = evaluate_cv()
    payload = cv_result_to_dict(result)
    payload["champion_reference_micro_adj"] = CHAMPION_REFERENCE_MICRO_ADJ
    payload["public_lb_best"] = PUBLIC_LB_BEST
    payload["cv_public_order_consistent"] = (
        result.micro_adj_edge_jaccard >= PUBLIC_LB_BEST
    )
    print(json.dumps(payload, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out}")

    if args.check_champion:
        delta = abs(result.micro_adj_edge_jaccard - CHAMPION_REFERENCE_MICRO_ADJ)
        if delta > args.tol:
            print(
                f"CHAMPION CV DRIFT: {result.micro_adj_edge_jaccard:.4f} != "
                f"{CHAMPION_REFERENCE_MICRO_ADJ} (delta {delta:.4f} > {args.tol})"
            )
            return 1
        print(
            f"champion CV reproduced: {result.micro_adj_edge_jaccard:.4f} "
            f"== {CHAMPION_REFERENCE_MICRO_ADJ} (delta {delta:.4f})"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
