"""Confirm the SOT-2369 short-track champion (min_track_length=4) end-to-end.

The screen (``screen_shorttrack.py``) re-links cached detections to sweep the
threshold cheaply. This confirm re-scores the promoted ``min_track_length=4``
through the **real champion code path** — ``biohub_tracking.champion.champion_params``
(reads ``champion/config.json``) → ``run_pipeline`` — on the same 4-dataset LB
holdout, so the promotion number comes from the exact detection+linking the
submission kernel will run, and it validates the config plumbing (that the new
``link.min_track_length`` key actually reaches ``LinkParams``).

Writes ``experiments/sot2369/confirm_shorttrack.json``. No Kaggle submission.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.eval.score import _jaccard, adjusted_edge_jaccard, evaluate
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff
from biohub_tracking.pipeline import run_pipeline

REPO = Path(__file__).resolve().parents[2]

HOLDOUT = [
    {"name": "44b6_0113de3b", "prefix": "44b6", "image": "data/test/44b6_0113de3b.zarr", "geff": "data/train/44b6_0113de3b.geff"},
    {"name": "44b6_0b24845f", "prefix": "44b6", "image": "data/test/44b6_0b24845f.zarr", "geff": "data/train/44b6_0b24845f.geff"},
    {"name": "6bba_05b6850b", "prefix": "6bba", "image": "data/test/6bba_05b6850b.zarr", "geff": "data/train/6bba_05b6850b.geff"},
    {"name": "6bba_05db0fb1", "prefix": "6bba", "image": "data/test/6bba_05db0fb1.zarr", "geff": "data/train/6bba_05db0fb1.geff"},
]

# Incumbent (SOT-2307 DoG-v3 adaptive) per-dataset adjusted edge Jaccard, for the
# side-by-side no-regression check (established champion numbers from registry.json).
INCUMBENT_ADJ = {
    "44b6_0113de3b": 0.8814, "44b6_0b24845f": 0.6658,
    "6bba_05b6850b": 0.5025, "6bba_05db0fb1": 0.7096,
}
INCUMBENT_MICRO = 0.6232


def main() -> None:
    config = load_champion_config()
    detect, link, _scale = champion_params(config)
    print(f"champion = {config['name']}  min_track_length={link.min_track_length}", flush=True)
    assert link.min_track_length == 4, "champion config did not carry min_track_length=4"

    rows = []
    for fam in HOLDOUT:
        print(f"running {fam['name']} ...", flush=True)
        geff = REPO / fam["geff"]
        gt = load_geff(geff)
        scale = geff_scale(geff)
        n_true = geff_estimated_num_nodes(geff)
        pred = run_pipeline(REPO / fam["image"], scale=scale, detect_params=detect, link_params=link)
        r = evaluate(pred, gt, scale=scale)
        j = _jaccard(r.edge_tp, r.edge_fp, r.edge_fn)
        adj = adjusted_edge_jaccard(j, r.num_pred_nodes, n_true)
        if adj != adj:
            adj = j
        row = {
            "name": fam["name"], "prefix": fam["prefix"],
            "edge_tp": r.edge_tp, "edge_fp": r.edge_fp, "edge_fn": r.edge_fn,
            "edge_jaccard": round(j, 4), "adjusted_edge_jaccard": round(adj, 4),
            "pred_nodes": r.num_pred_nodes, "weight": r.edge_tp + r.edge_fp + r.edge_fn,
            "incumbent_adj": INCUMBENT_ADJ[fam["name"]],
            "delta_vs_incumbent": round(adj - INCUMBENT_ADJ[fam["name"]], 4),
        }
        rows.append(row)
        print(f"  {row['name']:16s} tp/fp/fn={row['edge_tp']}/{row['edge_fp']}/{row['edge_fn']} "
              f"adj={row['adjusted_edge_jaccard']} (incumbent {row['incumbent_adj']}, "
              f"delta {row['delta_vs_incumbent']:+}) pred_nodes={row['pred_nodes']}", flush=True)

    tp = sum(r["edge_tp"] for r in rows); fp = sum(r["edge_fp"] for r in rows); fn = sum(r["edge_fn"] for r in rows)
    wsum = sum(r["weight"] * r["adjusted_edge_jaccard"] for r in rows if r["weight"] > 0)
    wtot = sum(r["weight"] for r in rows if r["weight"] > 0)
    micro = wsum / wtot if wtot > 0 else float("nan")
    by_prefix = {}
    for p in sorted({r["prefix"] for r in rows}):
        pr = [r for r in rows if r["prefix"] == p]
        w = sum(r["weight"] for r in pr)
        by_prefix[p] = round(sum(r["weight"] * r["adjusted_edge_jaccard"] for r in pr) / w, 4)

    no_regression = all(r["delta_vs_incumbent"] >= 0 for r in rows)
    verdict = "PROMOTE" if (micro > INCUMBENT_MICRO and no_regression) else "REJECT"
    result = {
        "issue": "SOT-2369",
        "title": "Confirm short-track champion (min_track_length=4) via the real champion pipeline",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "champion": config["name"],
        "holdout": [f["name"] for f in HOLDOUT],
        "per_dataset": rows,
        "holdout_micro_adj": round(micro, 4),
        "by_prefix": by_prefix,
        "incumbent_micro_adj": INCUMBENT_MICRO,
        "delta_micro_adj": round(micro - INCUMBENT_MICRO, 4),
        "no_per_dataset_regression": no_regression,
        "verdict": verdict,
        "reproducible_note": "Deterministic detect+link via champion_params(); re-running reproduces every score.",
    }
    print(f"\nholdout micro-adj={round(micro,4)} by_prefix={by_prefix} "
          f"(incumbent {INCUMBENT_MICRO}, delta {round(micro-INCUMBENT_MICRO,4):+})")
    print(f"no_per_dataset_regression={no_regression}  VERDICT: {verdict}")
    out = REPO / "experiments/sot2369/confirm_shorttrack.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
