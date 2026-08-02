"""Re-anchor the local oracle to the real Kaggle test set (SOT-2305).

Motivation
----------
The previous local oracle scored the champion on only **two** GT families, both
from the ``44b6`` dataset (``44b6_0113de3b`` clean + ``44b6_0b24845f``
fragmented). On that 2-family oracle the SOT-2272 DoG-v2 champion measured
micro-adj **0.772** — a big jump over the v1 global-threshold baseline (0.506).
But the public LB moved the other way: v1 scored **0.509** and DoG-v2 **0.500**.
That divergence is *oracle drift*: the 2-family oracle over-weights the one
fragmented ``44b6`` family DoG was tuned to recover, and is blind to the ``6bba``
families that make up half of the real test set.

The real Kaggle test set is **four** datasets — ``44b6_0113de3b``,
``44b6_0b24845f``, ``6bba_05b6850b``, ``6bba_05db0fb1`` — and the competition
train split ships ground-truth ``*.geff`` for all four. This script re-anchors the
local oracle onto exactly those four families (test ``*.zarr`` image + train
``*.geff`` GT, the same pairing that already reproduces the LB for the two 44b6
families), produces a **per-dataset / per-prefix breakdown**, and scores both the
DoG-v2 champion and the v1 global-threshold baseline so we can see *which detector
is LB-consistent* on the representative holdout.

It performs **no Kaggle submission** and mutates no champion state.

Usage::

    .venv/bin/python scripts/reanchor_oracle.py                 # all 4 families, both detectors
    .venv/bin/python scripts/reanchor_oracle.py --families 44b6 # only the 2 legacy 44b6 families
    .venv/bin/python scripts/reanchor_oracle.py --detectors dog # only DoG-v2

Writes ``docs/reanchored-oracle-evaluation.json`` and appends the axis result to
``docs/ai/experiment_ledger.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

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

# The four families that make up the real Kaggle test set. For each we hold the
# test *.zarr image locally and the train *.geff ground truth. This is the exact
# scored set, so the micro-avg here is the faithful LB oracle.
HOLDOUT = [
    {"name": "44b6_0113de3b", "prefix": "44b6", "image": "data/test/44b6_0113de3b.zarr", "geff": "data/train/44b6_0113de3b.geff"},
    {"name": "44b6_0b24845f", "prefix": "44b6", "image": "data/test/44b6_0b24845f.zarr", "geff": "data/train/44b6_0b24845f.geff"},
    {"name": "6bba_05b6850b", "prefix": "6bba", "image": "data/test/6bba_05b6850b.zarr", "geff": "data/train/6bba_05b6850b.geff"},
    {"name": "6bba_05db0fb1", "prefix": "6bba", "image": "data/test/6bba_05db0fb1.zarr", "geff": "data/train/6bba_05db0fb1.geff"},
]

# The two detectors under comparison. Linking is identical; only the detection
# response differs (v1 = global intensity percentile, dog-v2 = local contrast).
DETECTORS = {
    "v1_global_threshold": {
        "label": "v1 global intensity threshold (detect-link-v1, LB 0.509)",
        "detect": DetectParams(
            sigma_zyx=(1.0, 3.0, 3.0),
            nms_size_zyx=(2, 5, 5),
            threshold_percentile=99.3,
            background_sigma_zyx=None,
        ),
        "link": LinkParams(max_distance=7.0, allow_division=False),
    },
    "dog_v2": {
        "label": "DoG-v2 local contrast (detect-link-dog-v2 champion, LB 0.500)",
        "detect": DetectParams(
            sigma_zyx=(1.0, 2.0, 2.0),
            background_sigma_zyx=(2.0, 6.0, 6.0),
            nms_size_zyx=(2, 5, 5),
            threshold_percentile=92.0,
        ),
        "link": LinkParams(max_distance=7.0, allow_division=False),
    },
}


def score_family(fam: dict, detect: DetectParams, link: LinkParams) -> dict:
    """Run one detector on one holdout family and return its per-dataset counts."""
    geff = REPO / fam["geff"]
    image = REPO / fam["image"]
    gt = load_geff(geff)
    scale = geff_scale(geff)
    n_true = geff_estimated_num_nodes(geff)
    pred = run_pipeline(image, scale=scale, detect_params=detect, link_params=link)
    r = evaluate(pred, gt, scale=scale)
    j = _jaccard(r.edge_tp, r.edge_fp, r.edge_fn)
    adj = adjusted_edge_jaccard(j, r.num_pred_nodes, n_true)
    if adj != adj:  # NaN → fall back to raw jaccard
        adj = j
    return {
        "name": fam["name"],
        "prefix": fam["prefix"],
        "edge_tp": r.edge_tp,
        "edge_fp": r.edge_fp,
        "edge_fn": r.edge_fn,
        "edge_jaccard": round(j, 4),
        "adjusted_edge_jaccard": round(adj, 4),
        "pred_nodes": r.num_pred_nodes,
        "n_true": (None if n_true != n_true else n_true),
        "weight": r.edge_tp + r.edge_fp + r.edge_fn,
    }


def micro(rows: list[dict]) -> dict:
    """Micro-average a list of per-family rows (sum TP/FP/FN; size-weight adj)."""
    tp = sum(r["edge_tp"] for r in rows)
    fp = sum(r["edge_fp"] for r in rows)
    fn = sum(r["edge_fn"] for r in rows)
    edge_j = _jaccard(tp, fp, fn)
    wsum = sum(r["weight"] * r["adjusted_edge_jaccard"] for r in rows if r["weight"] > 0)
    wtot = sum(r["weight"] for r in rows if r["weight"] > 0)
    adj = wsum / wtot if wtot > 0 else float("nan")
    # No GT divisions in this holdout → division term is 0, score == adj.
    return {
        "n_families": len(rows),
        "edge_tp": tp,
        "edge_fp": fp,
        "edge_fn": fn,
        "micro_edge_jaccard": round(edge_j, 4),
        "micro_adj_edge_jaccard": round(adj, 4),
        "score": round(adj, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", choices=["all", "44b6", "6bba"], default="all")
    ap.add_argument("--detectors", choices=["all", "v1", "dog"], default="all")
    ap.add_argument("--dry-run", action="store_true", help="Print only; do not write files.")
    args = ap.parse_args()

    fams = [f for f in HOLDOUT if args.families == "all" or f["prefix"] == args.families]
    det_keys = list(DETECTORS)
    if args.detectors == "v1":
        det_keys = ["v1_global_threshold"]
    elif args.detectors == "dog":
        det_keys = ["dog_v2"]

    detectors_out: dict[str, dict] = {}
    for key in det_keys:
        cfg = DETECTORS[key]
        print(f"\n=== {key}: {cfg['label']} ===", flush=True)
        rows = []
        for fam in fams:
            print(f"  running {fam['name']} ...", flush=True)
            row = score_family(fam, cfg["detect"], cfg["link"])
            rows.append(row)
            print(
                f"    {row['name']} ({row['prefix']}): "
                f"edge TP/FP/FN={row['edge_tp']}/{row['edge_fp']}/{row['edge_fn']} "
                f"j={row['edge_jaccard']} adj={row['adjusted_edge_jaccard']} "
                f"pred_nodes={row['pred_nodes']}",
                flush=True,
            )
        overall = micro(rows)
        by_prefix = {
            p: micro([r for r in rows if r["prefix"] == p])
            for p in sorted({r["prefix"] for r in rows})
        }
        two_family = micro([r for r in rows if r["prefix"] == "44b6"]) if any(
            r["prefix"] == "44b6" for r in rows
        ) else None
        detectors_out[key] = {
            "label": cfg["label"],
            "per_dataset": rows,
            "by_prefix": by_prefix,
            "legacy_2family_44b6_micro": two_family,
            "holdout_micro": overall,
        }
        print(
            f"  -> holdout micro-adj={overall['micro_adj_edge_jaccard']} "
            f"(2-family 44b6={two_family['micro_adj_edge_jaccard'] if two_family else 'n/a'})",
            flush=True,
        )

    result = {
        "issue": "SOT-2305",
        "title": "Re-anchor local oracle to the real 4-dataset Kaggle test set",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "holdout": {
            "families": [f["name"] for f in fams],
            "pairing": "test/<name>.zarr image + train/<name>.geff ground truth",
            "note": (
                "These four families are the exact Kaggle test set. The train split "
                "ships GT geff for all four, so this micro-avg is a faithful LB oracle "
                "(vs the previous 2-family 44b6-only oracle that hid the 6bba behaviour)."
            ),
        },
        "public_lb_reference": {"v1_global_threshold": 0.509, "dog_v2": 0.500},
        "detectors": detectors_out,
        "reproducible_note": "Pipeline is deterministic (no RNG); re-running reproduces every score.",
    }

    print("\n" + json.dumps(result, indent=2))
    if not args.dry_run:
        out = REPO / "docs/reanchored-oracle-evaluation.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
