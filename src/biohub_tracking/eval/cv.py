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
# The champion forfeits divisions (allow_division=false → division_term 0.0), so
# the full re-anchored competition score equals the micro-adj byte-for-byte. Kept
# as a distinct constant so --check-champion can guard the *full-metric* number.
CHAMPION_REFERENCE_SCORE: float = 0.6649
PUBLIC_LB_BEST: float = 0.624  # champion public best, for CV↔public order check.

# Representativeness self-check tolerances (SOT-2817). ``MIX_SENSITIVITY_TOL`` is
# the gap between the sample-weighted micro and the family-mix-robust lineage
# macro beyond which the CV's *ranking* is flagged as family-mix sensitive — a
# change that helps the dominant lineage a hair while hurting the sparse one can
# raise the micro but lower the LB (SOT-2816 §5). ``MAGNITUDE_RATIO_BOUND`` is the
# order-of-magnitude ("桁一致") sanity band for CV vs the public LB.
MIX_SENSITIVITY_TOL: float = 0.05
MAGNITUDE_RATIO_BOUND: float = 2.0


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
    """Aggregated CV: full competition metric + representativeness breakdowns.

    Re-anchored (SOT-2817) so the reported ``score`` is the *complete*
    competition metric ``adjusted_edge_jaccard + 0.1 · division_jaccard`` on
    every run, and the division contribution is always explicit — including
    ``division_term = 0.0`` when the pipeline predicts no forks (the champion's
    ``allow_division=false`` case) or when the holdout GT has no division events,
    so the 0.1 term never silently vanishes as ``null``.

    Representativeness guards (the micro is 95.8% 6bba-weighted on this holdout,
    SOT-2816 §5) add a family-mix-robust view alongside the sample-weighted micro:

    * ``macro_adj_edge_jaccard`` — unweighted mean over families (each embryo
      video counts equally, so a dense family's raw edge count cannot dominate).
    * ``lineage_macro_adj`` — mean over the two *lineages* of each lineage's
      weighted micro, so the sparse 44b6 lineage (4.2% of the micro weight)
      contributes at parity with the dense 6bba lineage.
    * ``by_lineage_weight_share`` — the share of the micro weight each lineage
      carries, so the domination is reported, not hidden.
    """

    micro_adj_edge_jaccard: float
    micro_edge_jaccard: float
    macro_adj_edge_jaccard: float
    lineage_macro_adj: float
    division_jaccard: float
    division_measurable: bool
    division_term: float
    score: float
    per_dataset: tuple[FamilyResult, ...]
    by_lineage: dict[str, float]
    by_lineage_weight_share: dict[str, float]

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

    Re-anchored (SOT-2817): the ``score`` is always the full competition metric
    ``micro_adj + 0.1 · division_jaccard`` with an **explicit** ``division_term``
    (``0.0`` when no fork is predicted or no GT division event exists), and the
    result carries the family-mix-robust macro views. None of this changes the
    champion number: with the champion forfeiting divisions the division term is
    ``0.0`` and ``score == micro_adj == 0.6649`` byte-for-byte.
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

    lineages = sorted({r.lineage for r in rows})
    by_lineage = {
        lin: _weighted_adj(tuple(r for r in rows if r.lineage == lin))
        for lin in lineages
    }

    # Representativeness guards (SOT-2817): a family-mix-robust view alongside
    # the sample-weighted micro, which on this holdout is 95.8% 6bba-weighted.
    scored = [r for r in rows if not math.isnan(r.adj_edge_jaccard)]
    macro_adj = (
        sum(r.adj_edge_jaccard for r in scored) / len(scored)
        if scored
        else float("nan")
    )
    lineage_vals = [v for v in by_lineage.values() if not math.isnan(v)]
    lineage_macro_adj = (
        sum(lineage_vals) / len(lineage_vals) if lineage_vals else float("nan")
    )
    total_weight = sum(r.weight for r in rows)
    by_lineage_weight_share = {
        lin: (
            sum(r.weight for r in rows if r.lineage == lin) / total_weight
            if total_weight > 0
            else float("nan")
        )
        for lin in lineages
    }

    # Full competition metric, division term always explicit. It is measurable
    # only when the confusion matrix is non-empty (a GT division event exists or
    # a fork was predicted); otherwise the term contributes exactly 0.0 and the
    # Jaccard is undefined (NaN) rather than silently dropped.
    division_measurable = (div_tp + div_fp + div_fn) > 0
    if division_measurable:
        division_jaccard = _jaccard(div_tp, div_fp, div_fn)
        division_term = SCORE_DIVISION_WEIGHT * division_jaccard
    else:
        division_jaccard = float("nan")
        division_term = 0.0
    score = micro_adj + division_term

    return CvResult(
        micro_adj_edge_jaccard=micro_adj,
        micro_edge_jaccard=micro_edge_jaccard,
        macro_adj_edge_jaccard=macro_adj,
        lineage_macro_adj=lineage_macro_adj,
        division_jaccard=division_jaccard,
        division_measurable=division_measurable,
        division_term=division_term,
        score=score,
        per_dataset=rows,
        by_lineage=by_lineage,
        by_lineage_weight_share=by_lineage_weight_share,
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


def representativeness_report(
    result: CvResult,
    *,
    public_lb_best: float = PUBLIC_LB_BEST,
    mix_tol: float = MIX_SENSITIVITY_TOL,
    magnitude_bound: float = MAGNITUDE_RATIO_BOUND,
) -> dict:
    """Self-check whether the CV micro is a faithful LB ranker (SOT-2817).

    Pure function of a :class:`CvResult` — no disk, no data. It surfaces the
    representativeness limits the SOT-2816 audit measured so a promotion never
    trusts a family-mix-sensitive micro blindly:

    * ``dominant_lineage`` / ``dominant_lineage_weight_share`` — which lineage
      carries the micro (95.8% 6bba on this holdout) and by how much.
    * ``micro_lineage_macro_gap`` / ``family_mix_sensitive`` — the gap between the
      sample-weighted micro and the parity-weighted lineage macro; when it
      exceeds ``mix_tol`` the CV ranking is family-mix sensitive, so a bare micro
      gain is *not* sufficient evidence for promotion (gate on macro too).
    * ``cv_public_order_consistent`` / ``cv_public_same_magnitude`` — the micro is
      not below and is within one order of magnitude of the public LB best (a
      CV that is a decimal order off the LB is not measuring the same thing).
    """
    micro = result.micro_adj_edge_jaccard
    share = result.by_lineage_weight_share
    if share:
        dominant_lineage = max(share, key=lambda k: share[k])
        dominant_share = share[dominant_lineage]
    else:  # pragma: no cover - aggregate always yields at least one lineage
        dominant_lineage, dominant_share = None, float("nan")

    lineage_macro = result.lineage_macro_adj
    mix_gap = (
        abs(micro - lineage_macro) if not math.isnan(lineage_macro) else float("nan")
    )
    family_mix_sensitive = (not math.isnan(mix_gap)) and mix_gap > mix_tol

    order_consistent = (not math.isnan(micro)) and micro >= public_lb_best
    if math.isnan(micro) or public_lb_best <= 0:
        same_magnitude = False
    else:
        ratio = micro / public_lb_best
        same_magnitude = (1.0 / magnitude_bound) <= ratio <= magnitude_bound

    return {
        "dominant_lineage": dominant_lineage,
        "dominant_lineage_weight_share": _round_or_none(dominant_share),
        "micro_lineage_macro_gap": _round_or_none(mix_gap),
        "family_mix_sensitive": family_mix_sensitive,
        "public_lb_best": public_lb_best,
        "cv_public_order_consistent": order_consistent,
        "cv_public_same_magnitude": same_magnitude,
    }


def _round_or_none(value: float, ndigits: int = 4) -> float | None:
    """Round *value*, mapping NaN to ``None`` for JSON."""
    return None if math.isnan(value) else round(value, ndigits)


def cv_result_to_dict(result: CvResult) -> dict:
    """JSON-serialisable view of a :class:`CvResult` (rounded for reports)."""
    return {
        "micro_adj_edge_jaccard": round(result.micro_adj_edge_jaccard, 4),
        "micro_edge_jaccard": round(result.micro_edge_jaccard, 4),
        "macro_adj_edge_jaccard": _round_or_none(result.macro_adj_edge_jaccard),
        "lineage_macro_adj": _round_or_none(result.lineage_macro_adj),
        # Full competition metric, division term always explicit (never dropped).
        "division_jaccard": _round_or_none(result.division_jaccard),
        "division_measurable": result.division_measurable,
        "division_term": round(result.division_term, 4),
        "score": round(result.score, 4),
        "by_lineage": {k: round(v, 4) for k, v in result.by_lineage.items()},
        "by_lineage_weight_share": {
            k: _round_or_none(v) for k, v in result.by_lineage_weight_share.items()
        },
        "representativeness": representativeness_report(result),
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
    payload["champion_reference_score"] = CHAMPION_REFERENCE_SCORE
    print(json.dumps(payload, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out}")

    if args.check_champion:
        # Guard both the micro-adj AND the full re-anchored competition score
        # (they coincide for the division-forfeiting champion, but checking the
        # score too pins the division-term wiring that this issue re-anchored).
        delta_micro = abs(result.micro_adj_edge_jaccard - CHAMPION_REFERENCE_MICRO_ADJ)
        delta_score = abs(result.score - CHAMPION_REFERENCE_SCORE)
        if delta_micro > args.tol or delta_score > args.tol:
            print(
                f"CHAMPION CV DRIFT: micro-adj {result.micro_adj_edge_jaccard:.4f} vs "
                f"{CHAMPION_REFERENCE_MICRO_ADJ} (delta {delta_micro:.4f}); "
                f"score {result.score:.4f} vs {CHAMPION_REFERENCE_SCORE} "
                f"(delta {delta_score:.4f}); tol {args.tol}"
            )
            return 1
        print(
            f"champion CV reproduced: micro-adj {result.micro_adj_edge_jaccard:.4f} "
            f"== {CHAMPION_REFERENCE_MICRO_ADJ} (delta {delta_micro:.4f}); "
            f"full score {result.score:.4f} == {CHAMPION_REFERENCE_SCORE} "
            f"(delta {delta_score:.4f}, division_term {result.division_term:.4f})"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
