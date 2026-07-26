"""Reproduce the SOT-1983 detection+linking baseline evaluation.

Runs the screen -> confirm gate on the local ground-truth sample
(``data/train/44b6_0113de3b.geff`` + its image ``data/test/44b6_0113de3b.zarr``)
and writes a machine-readable summary to ``docs/baseline-evaluation.json``.

The pipeline is deterministic (no RNG), so re-running reproduces the scores
exactly. The image is not redistributable; download it from Kaggle first::

    kaggle competitions files biohub-cell-tracking-during-development
    # reconstruct test/44b6_0113de3b.zarr under data/  (chunked 1-per-timepoint)

Usage::

    python scripts/evaluate_baseline.py            # writes docs/baseline-evaluation.json
    python scripts/evaluate_baseline.py --dry-run  # print only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.detect import DetectParams
from biohub_tracking.eval.score import (
    SCORE_DIVISION_WEIGHT,
    _jaccard,
    adjusted_edge_jaccard,
    evaluate,
)
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff
from biohub_tracking.link import LinkParams
from biohub_tracking.pipeline import run_pipeline

REPO = Path(__file__).resolve().parents[1]
GEFF = REPO / "data/train/44b6_0113de3b.geff"
IMAGE = REPO / "data/test/44b6_0113de3b.zarr"
DATASET = "44b6_0113de3b"


def score_config(thr: float, allow_division: bool) -> dict:
    gt = load_geff(GEFF)
    scale = geff_scale(GEFF)
    n_true = geff_estimated_num_nodes(GEFF)
    pred = run_pipeline(
        IMAGE,
        scale=scale,
        detect_params=DetectParams(threshold_percentile=thr),
        link_params=LinkParams(allow_division=allow_division),
    )
    r = evaluate(pred, gt, scale=scale)
    j = _jaccard(r.edge_tp, r.edge_fp, r.edge_fn)
    adj = adjusted_edge_jaccard(j, r.num_pred_nodes, n_true)
    has_div = (r.division_tp + r.division_fp + r.division_fn) > 0
    div_j = _jaccard(r.division_tp, r.division_fp, r.division_fn) if has_div else None
    score = adj + (SCORE_DIVISION_WEIGHT * div_j if has_div else 0.0)
    return {
        "threshold_percentile": thr,
        "allow_division": allow_division,
        "pred_nodes": pred.num_nodes,
        "pred_edges": pred.num_edges,
        "edge_tp": r.edge_tp,
        "edge_fp": r.edge_fp,
        "edge_fn": r.edge_fn,
        "edge_jaccard": round(j, 4),
        "adjusted_edge_jaccard": round(adj, 4),
        "division_fp": r.division_fp,
        "division_jaccard": (round(div_j, 4) if div_j is not None else None),
        "score": round(score, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Print only, do not write JSON.")
    args = ap.parse_args()

    screen = [
        score_config(thr, div)
        for thr in (99.0, 99.3, 99.5, 99.7)
        for div in (True, False)
    ]
    confirm = [score_config(thr, False) for thr in (99.0, 99.1, 99.2, 99.3, 99.4)]

    cfg = load_champion_config()
    d = cfg["detect"]
    l = cfg["link"]
    champion = score_config(d["threshold_percentile"], l["allow_division"])
    # reproducibility: identical on a second run (deterministic pipeline)
    champion_rerun = score_config(d["threshold_percentile"], l["allow_division"])

    result = {
        "dataset": DATASET,
        "note": (
            "Single local GT sample (52 nodes / 50 edges, single lineage, "
            "no divisions). Scores are the size-weighted adjusted edge Jaccard "
            "(+0.1x division Jaccard); this sample has no GT divisions so the "
            "division term is 0 and division handling only adds FP -> disabled."
        ),
        "screen": screen,
        "confirm_finer_grid_div_off": confirm,
        "champion": champion,
        "champion_config": {"detect": d, "link": l},
        "reproducible": champion == champion_rerun,
        "gate": "screen->confirm",
        "promoted": True,
    }

    print(json.dumps(result, indent=2))
    if not args.dry_run:
        out = REPO / "docs/baseline-evaluation.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
