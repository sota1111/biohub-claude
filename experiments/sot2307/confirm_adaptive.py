"""Independent confirm of the SOT-2307 adaptive-threshold winner (mad_k=3.0).

The screen (``screen_adaptive.py``) re-thresholds a cached candidate set to sweep
``k`` cheaply. This confirm re-scores the winning ``mad_k=3.0`` through the **real
production detector** — ``biohub_tracking.pipeline.run_pipeline`` with
``DetectParams(mad_k=3.0)`` — on the same 4-dataset holdout, so the promotion number
comes from the exact code path a champion would run, not the sweep shortcut. It also
re-scores the incumbent DoG-v2 (percentile 92) with the identical harness so the A/B
is measured side-by-side in one run (rejects screen upward bias).

Writes ``experiments/sot2307/confirm_adaptive.json``. No Kaggle submission.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.detect import DetectParams
from biohub_tracking.eval.score import _jaccard, adjusted_edge_jaccard, evaluate
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff
from biohub_tracking.link import LinkParams
from biohub_tracking.pipeline import run_pipeline

REPO = Path(__file__).resolve().parents[2]

HOLDOUT = [
    {"name": "44b6_0113de3b", "prefix": "44b6", "image": "data/test/44b6_0113de3b.zarr", "geff": "data/train/44b6_0113de3b.geff"},
    {"name": "44b6_0b24845f", "prefix": "44b6", "image": "data/test/44b6_0b24845f.zarr", "geff": "data/train/44b6_0b24845f.geff"},
    {"name": "6bba_05b6850b", "prefix": "6bba", "image": "data/test/6bba_05b6850b.zarr", "geff": "data/train/6bba_05b6850b.geff"},
    {"name": "6bba_05db0fb1", "prefix": "6bba", "image": "data/test/6bba_05db0fb1.zarr", "geff": "data/train/6bba_05db0fb1.geff"},
]

LINK = LinkParams(max_distance=7.0, allow_division=False)

CANDIDATES = {
    "incumbent_dog_v2_percentile92": DetectParams(
        sigma_zyx=(1.0, 2.0, 2.0),
        background_sigma_zyx=(2.0, 6.0, 6.0),
        nms_size_zyx=(2, 5, 5),
        threshold_percentile=92.0,
    ),
    "adaptive_mad_k3": DetectParams(
        sigma_zyx=(1.0, 2.0, 2.0),
        background_sigma_zyx=(2.0, 6.0, 6.0),
        nms_size_zyx=(2, 5, 5),
        mad_k=3.0,
    ),
}


def score_family(fam: dict, detect: DetectParams) -> dict:
    geff = REPO / fam["geff"]
    gt = load_geff(geff)
    scale = geff_scale(geff)
    n_true = geff_estimated_num_nodes(geff)
    pred = run_pipeline(REPO / fam["image"], scale=scale, detect_params=detect, link_params=LINK)
    r = evaluate(pred, gt, scale=scale)
    j = _jaccard(r.edge_tp, r.edge_fp, r.edge_fn)
    adj = adjusted_edge_jaccard(j, r.num_pred_nodes, n_true)
    if adj != adj:
        adj = j
    # precision/recall on matched edges, for the "no per-dataset divergence" check.
    prec = r.edge_tp / (r.edge_tp + r.edge_fp) if (r.edge_tp + r.edge_fp) else float("nan")
    rec = r.edge_tp / (r.edge_tp + r.edge_fn) if (r.edge_tp + r.edge_fn) else float("nan")
    return {
        "name": fam["name"], "prefix": fam["prefix"],
        "edge_tp": r.edge_tp, "edge_fp": r.edge_fp, "edge_fn": r.edge_fn,
        "edge_jaccard": round(j, 4), "adjusted_edge_jaccard": round(adj, 4),
        "edge_precision": round(prec, 4), "edge_recall": round(rec, 4),
        "pred_nodes": r.num_pred_nodes,
        "n_true": (None if n_true != n_true else n_true),
        "weight": r.edge_tp + r.edge_fp + r.edge_fn,
    }


def micro(rows: list[dict]) -> dict:
    tp = sum(r["edge_tp"] for r in rows); fp = sum(r["edge_fp"] for r in rows); fn = sum(r["edge_fn"] for r in rows)
    wsum = sum(r["weight"] * r["adjusted_edge_jaccard"] for r in rows if r["weight"] > 0)
    wtot = sum(r["weight"] for r in rows if r["weight"] > 0)
    adj = wsum / wtot if wtot > 0 else float("nan")
    return {"edge_tp": tp, "edge_fp": fp, "edge_fn": fn,
            "micro_edge_jaccard": round(_jaccard(tp, fp, fn), 4),
            "micro_adj_edge_jaccard": round(adj, 4), "score": round(adj, 4)}


def main() -> None:
    out_detectors = {}
    for key, detect in CANDIDATES.items():
        print(f"\n=== {key} (real run_pipeline) ===", flush=True)
        rows = []
        for fam in HOLDOUT:
            print(f"  running {fam['name']} ...", flush=True)
            row = score_family(fam, detect)
            rows.append(row)
            print(f"    {row['name']:16s} tp/fp/fn={row['edge_tp']}/{row['edge_fp']}/{row['edge_fn']} "
                  f"adj={row['adjusted_edge_jaccard']} prec={row['edge_precision']} rec={row['edge_recall']} "
                  f"pred_nodes={row['pred_nodes']}", flush=True)
        overall = micro(rows)
        by_prefix = {p: micro([r for r in rows if r["prefix"] == p]) for p in sorted({r["prefix"] for r in rows})}
        out_detectors[key] = {"per_dataset": rows,
                              "by_prefix": {p: v["score"] for p, v in by_prefix.items()},
                              "holdout_micro": overall}
        print(f"  -> holdout micro-adj={overall['micro_adj_edge_jaccard']} "
              f"by_prefix={{ {', '.join(f'{p}:{v}' for p, v in out_detectors[key]['by_prefix'].items())} }}", flush=True)

    inc = out_detectors["incumbent_dog_v2_percentile92"]["holdout_micro"]["score"]
    adp = out_detectors["adaptive_mad_k3"]["holdout_micro"]["score"]
    verdict = "PROMOTE" if adp > inc else "REJECT"
    result = {
        "issue": "SOT-2307",
        "title": "Confirm adaptive mad_k=3 vs incumbent DoG-v2 via the real pipeline",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "holdout": [f["name"] for f in HOLDOUT],
        "detectors": out_detectors,
        "delta_micro_adj": round(adp - inc, 4),
        "verdict": verdict,
        "reproducible_note": "Deterministic run_pipeline; re-running reproduces every score.",
    }
    print(f"\nVERDICT: {verdict}  incumbent={inc}  adaptive_mad_k3={adp}  delta={round(adp - inc, 4)}")
    out = REPO / "experiments/sot2307/confirm_adaptive.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
