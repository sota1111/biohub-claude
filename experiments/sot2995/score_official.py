"""SOT-2995 step 2 (official venv): re-score saved predictions with the GENUINE
royerlab scorer and measure divergence vs the clean-room oracle.

Reads the ``.geff`` predictions written by ``predict.py`` plus the ground-truth
geffs in ``data/train``, runs both the official and the clean-room scorers on
each, and writes per-family divergence + official per-family FamilyResult rows.

Run with the OFFICIAL venv (tracksdata + tracking_cellmot installed) and the repo
``src`` on PYTHONPATH:
    PYTHONPATH=src /path/to/official-venv/bin/python \
        experiments/sot2995/score_official.py --label champion [--label v1 ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biohub_tracking.eval.cv import CV_HOLDOUT
from biohub_tracking.eval.official import (
    divergence_row,
    divergence_row_to_dict,
    official_available,
    official_score_family,
)
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = Path("/tmp/sot2995")
_FAM_BY_NAME = {f.name: f for f in CV_HOLDOUT}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", required=True)
    args = ap.parse_args()

    if not official_available():
        raise SystemExit("official scorer unavailable in this interpreter")

    for label in args.label:
        pred_dir = OUT_ROOT / label
        official_rows = []
        divergence = []
        for fam in CV_HOLDOUT:
            pred_geff = pred_dir / f"{fam.name}.geff"
            if not pred_geff.exists():
                continue
            gt_geff = REPO / fam.geff
            pred = load_geff(pred_geff)
            gt = load_geff(gt_geff)
            scale = geff_scale(gt_geff)
            n_true = geff_estimated_num_nodes(gt_geff)

            fr = official_score_family(
                fam.name, fam.lineage, pred, gt, n_true, scale=scale
            )
            official_rows.append(
                {
                    "name": fr.name,
                    "lineage": fr.lineage,
                    "edge_tp": fr.edge_tp,
                    "edge_fp": fr.edge_fp,
                    "edge_fn": fr.edge_fn,
                    "edge_jaccard": fr.edge_jaccard,
                    "adj_edge_jaccard": fr.adj_edge_jaccard,
                    "division_tp": fr.division_tp,
                    "division_fp": fr.division_fp,
                    "division_fn": fr.division_fn,
                    "num_pred_nodes": fr.num_pred_nodes,
                    "n_true": fr.n_true,
                    "weight": fr.weight,
                }
            )
            drow = divergence_row(fam.name, pred, gt, scale=scale)
            divergence.append(divergence_row_to_dict(drow))
            print(
                f"[{label}] {fam.name}: official edge "
                f"{fr.edge_tp}/{fr.edge_fp}/{fr.edge_fn} "
                f"div {fr.division_tp}/{fr.division_fp}/{fr.division_fn} "
                f"| counts_match={drow.counts_match}",
                flush=True,
            )

        out = pred_dir / "official_scores.json"
        out.write_text(
            json.dumps(
                {"label": label, "rows": official_rows, "divergence": divergence},
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
