"""Screen short-track pruning (SOT-2369) on the SOT-2305 4-dataset LB holdout.

Ports the reference tracker's ``FILTER_SHORT_TRACKS`` post-processing into the
classical CPU linker: after linking, drop weakly-connected track fragments with
fewer than ``min_track_length`` nodes. A real cell persists across many frames,
so a detection that never links into a multi-frame track is almost always noise.
Dropping singletons removes nodes that carry no matched edge, so edge TP/FP/FN
are untouched while the predicted node count falls — a strict adjusted-Jaccard
gain whenever the champion over-predicts (the node-count penalty is what caps the
score on the sparse 6bba families).

Detection is frozen (champion DoG-v3 adaptive ``mad_k=3.0``) and computed **once**
per dataset; each ``min_track_length`` is re-linked off the cached detections.
Scored through the same evaluator the champion was promoted on.

Writes ``experiments/sot2369/screen_shorttrack.json``. No Kaggle submission.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.detect import DetectParams, detect_volume_series
from biohub_tracking.eval.score import _jaccard, adjusted_edge_jaccard, evaluate
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff
from biohub_tracking.link import LinkParams, link_centroids
from biohub_tracking.pipeline import _open_image_array

REPO = Path(__file__).resolve().parents[2]

HOLDOUT = [
    {"name": "44b6_0113de3b", "prefix": "44b6", "image": "data/test/44b6_0113de3b.zarr", "geff": "data/train/44b6_0113de3b.geff"},
    {"name": "44b6_0b24845f", "prefix": "44b6", "image": "data/test/44b6_0b24845f.zarr", "geff": "data/train/44b6_0b24845f.geff"},
    {"name": "6bba_05b6850b", "prefix": "6bba", "image": "data/test/6bba_05b6850b.zarr", "geff": "data/train/6bba_05b6850b.geff"},
    {"name": "6bba_05db0fb1", "prefix": "6bba", "image": "data/test/6bba_05db0fb1.zarr", "geff": "data/train/6bba_05db0fb1.geff"},
]

DETECT = DetectParams(
    sigma_zyx=(1.0, 2.0, 2.0),
    background_sigma_zyx=(2.0, 6.0, 6.0),
    nms_size_zyx=(2, 5, 5),
    mad_k=3.0,
)

MIN_TRACK_LENGTHS = [1, 2, 3, 4, 5]  # 1 == champion (keep everything)


def variant(mtl: int) -> LinkParams:
    return LinkParams(max_distance=7.0, allow_division=False, min_track_length=mtl)


def score_from_graph(pred, geff: Path, scale, n_true, name, prefix) -> dict:
    gt = load_geff(geff)
    r = evaluate(pred, gt, scale=scale)
    j = _jaccard(r.edge_tp, r.edge_fp, r.edge_fn)
    adj = adjusted_edge_jaccard(j, r.num_pred_nodes, n_true)
    if adj != adj:
        adj = j
    return {
        "name": name, "prefix": prefix,
        "edge_tp": r.edge_tp, "edge_fp": r.edge_fp, "edge_fn": r.edge_fn,
        "edge_jaccard": round(j, 4), "adjusted_edge_jaccard": round(adj, 4),
        "pred_nodes": r.num_pred_nodes, "n_true": (None if n_true != n_true else n_true),
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
    cached = {}
    for fam in HOLDOUT:
        print(f"detecting {fam['name']} ...", flush=True)
        geff = REPO / fam["geff"]
        scale = geff_scale(geff)
        n_true = geff_estimated_num_nodes(geff)
        arr = _open_image_array(REPO / fam["image"])
        detections = detect_volume_series(arr, DETECT)
        cached[fam["name"]] = {"detections": detections, "scale": scale, "n_true": n_true,
                               "prefix": fam["prefix"], "geff": geff}
        total = sum(len(v) for v in detections.values())
        print(f"  {fam['name']}: {total} detections over {len(detections)} timepoints", flush=True)

    out_variants = {}
    for mtl in MIN_TRACK_LENGTHS:
        key = f"min_track_len_{mtl}"
        print(f"\n=== {key} ===", flush=True)
        rows = []
        for name, c in cached.items():
            pred = link_centroids(c["detections"], scale=c["scale"], params=variant(mtl))
            row = score_from_graph(pred, c["geff"], c["scale"], c["n_true"], name, c["prefix"])
            rows.append(row)
            print(f"  {name:16s} tp/fp/fn={row['edge_tp']}/{row['edge_fp']}/{row['edge_fn']} "
                  f"adj={row['adjusted_edge_jaccard']} pred_nodes={row['pred_nodes']} (n_true={row['n_true']})", flush=True)
        overall = micro(rows)
        by_prefix = {p: micro([r for r in rows if r["prefix"] == p]) for p in sorted({r["prefix"] for r in rows})}
        out_variants[key] = {"per_dataset": rows,
                             "by_prefix": {p: v["score"] for p, v in by_prefix.items()},
                             "holdout_micro": overall}
        print(f"  -> holdout micro-adj={overall['micro_adj_edge_jaccard']} by_prefix={out_variants[key]['by_prefix']}", flush=True)

    inc = out_variants["min_track_len_1"]["holdout_micro"]["score"]
    ranked = sorted(((k, v["holdout_micro"]["score"]) for k, v in out_variants.items()), key=lambda kv: -kv[1])
    best_key, best_score = ranked[0]
    verdict = "PROMOTE" if best_key != "min_track_len_1" and best_score > inc else "REJECT"
    result = {
        "issue": "SOT-2369",
        "title": "Screen short-track pruning (FILTER_SHORT_TRACKS) vs champion linker",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "holdout": [f["name"] for f in HOLDOUT],
        "detector": "champion DoG-v3 adaptive mad_k=3.0 (frozen)",
        "variants": out_variants,
        "incumbent_score": inc,
        "ranked": ranked,
        "best": {"variant": best_key, "score": best_score, "delta_vs_incumbent": round(best_score - inc, 4)},
        "verdict": verdict,
        "reproducible_note": "Deterministic detect+link; re-running reproduces every score.",
    }
    print(f"\nVERDICT: {verdict}  incumbent(mtl=1)={inc}  best={best_key}:{best_score}  delta={round(best_score - inc, 4)}")
    out = REPO / "experiments/sot2369/screen_shorttrack.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
