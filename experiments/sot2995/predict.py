"""SOT-2995 step 1 (repo .venv): predict the CV holdout for a config and save.

Runs the deterministic detect+link pipeline over each leak-free CV family for a
given champion-style config, writes each predicted graph to a ``.geff`` the
official scorer can read, and records the clean-room per-family score. Kept out
of the light package on purpose (touches disk + the pipeline).

Usage:
    python experiments/sot2995/predict.py --config <config.json> --label <name> \
        [--only <family>]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from biohub_tracking.champion import (
    champion_params,
    learned_detector_config,
)
from biohub_tracking.eval.cv import CV_HOLDOUT, score_family
from biohub_tracking.eval.official import graph_to_geff
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff
from biohub_tracking.pipeline import run_pipeline

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = Path("/tmp/sot2995")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--only", default=None, help="Score just this family (timing).")
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text())
    detect, link, _scale = champion_params(config)
    learned_detector = None
    ld_block = learned_detector_config(config)
    if ld_block is not None:
        from biohub_tracking.learned_detect import build_learned_detector

        learned_detector = build_learned_detector(ld_block)

    out_dir = OUT_ROOT / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_rows = []
    for fam in CV_HOLDOUT:
        if args.only and fam.name != args.only:
            continue
        geff = REPO / fam.geff
        image = REPO / fam.image
        gt = load_geff(geff)
        scale = geff_scale(geff)
        n_true = geff_estimated_num_nodes(geff)

        t0 = time.time()
        pred = run_pipeline(
            image,
            scale=scale,
            detect_params=detect,
            link_params=link,
            learned_detector=learned_detector,
        )
        dt = time.time() - t0

        graph_to_geff(
            pred,
            out_dir / f"{fam.name}.geff",
            scale=scale,
            estimated_number_of_nodes=n_true,
        )
        fr = score_family(fam, pred, gt, n_true, scale=scale)
        clean_rows.append(
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
        print(
            f"{fam.name}: pred_nodes={pred.num_nodes} edges={pred.num_edges} "
            f"edge {fr.edge_tp}/{fr.edge_fp}/{fr.edge_fn} adj={fr.adj_edge_jaccard:.4f} "
            f"({dt:.1f}s)",
            flush=True,
        )

    (out_dir / "clean_scores.json").write_text(
        json.dumps({"label": args.label, "rows": clean_rows}, indent=2) + "\n"
    )
    print(f"wrote {out_dir/'clean_scores.json'} ({len(clean_rows)} families)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
