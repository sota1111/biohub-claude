"""Command-line local evaluator.

Score a submission CSV against a directory of ground-truth ``*.geff`` files::

    python -m biohub_tracking.evaluate --pred submission.csv --gt-dir data/train

Prints per-dataset edge/division counts and the micro-averaged combined score
(``adjusted_edge_jaccard + 0.1 · division_jaccard``), the same metric the Kaggle
leaderboard uses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .eval.score import (
    SCORE_DIVISION_WEIGHT,
    DatasetsResult,
    _jaccard,
    adjusted_edge_jaccard,
    evaluate,
)
from .io import geff_estimated_num_nodes, geff_scale, load_geff, load_submission_csv
from .matching import DEFAULT_MAX_DISTANCE


def evaluate_submission(
    pred_csv: Path | str,
    gt_dir: Path | str,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    verbose: bool = True,
) -> DatasetsResult:
    """Evaluate *pred_csv* against every geff under *gt_dir* sharing a dataset name."""
    pred_graphs = load_submission_csv(pred_csv)
    gt_dir = Path(gt_dir)

    edge_tp = edge_fp = edge_fn = 0
    div_tp = div_fp = div_fn = 0
    weighted_adj = 0.0
    total_w = 0.0

    geffs = sorted(gt_dir.glob("*.geff"))
    if not geffs:
        raise FileNotFoundError(f"no *.geff files found under {gt_dir}")

    for geff in geffs:
        name = geff.stem
        if name not in pred_graphs:
            if verbose:
                print(f"{name}: MISSING from submission — all GT edges counted as FN")
        gt = load_geff(geff)
        pred = pred_graphs.get(name, type(gt)())
        scale = geff_scale(geff)
        n_true = geff_estimated_num_nodes(geff)

        r = evaluate(pred, gt, scale=scale, max_distance=max_distance)
        edge_tp += r.edge_tp
        edge_fp += r.edge_fp
        edge_fn += r.edge_fn
        div_tp += r.division_tp
        div_fp += r.division_fp
        div_fn += r.division_fn

        j = _jaccard(r.edge_tp, r.edge_fp, r.edge_fn)
        adj = adjusted_edge_jaccard(j, r.num_pred_nodes, n_true)
        if adj != adj:  # NaN → fall back to raw jaccard
            adj = j
        w = r.edge_tp + r.edge_fp + r.edge_fn
        if w > 0 and adj == adj:
            weighted_adj += w * adj
            total_w += w

        if verbose:
            print(
                f"{name}: edge TP/FP/FN={r.edge_tp}/{r.edge_fp}/{r.edge_fn} "
                f"jaccard={j:.4f} adj={adj:.4f} | "
                f"div TP/FP/FN={r.division_tp}/{r.division_fp}/{r.division_fn} | "
                f"pred_nodes={r.num_pred_nodes} n_true={n_true}"
            )

    edge_jaccard = _jaccard(edge_tp, edge_fp, edge_fn)
    adj_edge_jaccard = weighted_adj / total_w if total_w > 0 else float("nan")
    has_div = (div_tp + div_fp + div_fn) > 0
    division_jaccard = _jaccard(div_tp, div_fp, div_fn) if has_div else float("nan")
    score = (
        adj_edge_jaccard + SCORE_DIVISION_WEIGHT * division_jaccard
        if has_div
        else adj_edge_jaccard
    )

    result = DatasetsResult(
        edge_jaccard=edge_jaccard,
        division_jaccard=division_jaccard,
        adj_edge_jaccard=adj_edge_jaccard,
        score=score,
    )
    if verbose:
        print("-" * 60)
        print(f"edge_jaccard      = {edge_jaccard:.4f}  (micro-avg)")
        print(f"adj_edge_jaccard  = {adj_edge_jaccard:.4f}  (size-weighted)")
        print(f"division_jaccard  = {division_jaccard:.4f}  (micro-avg)")
        print(f"SCORE             = {score:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True, help="Submission CSV path.")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Directory of GT *.geff files.")
    parser.add_argument("--max-distance", type=float, default=DEFAULT_MAX_DISTANCE)
    args = parser.parse_args()
    evaluate_submission(args.pred, args.gt_dir, max_distance=args.max_distance)


if __name__ == "__main__":
    main()
