"""Screen the SOT-2307 adaptive-intensity DoG threshold on the 4-dataset holdout.

Axis (SOT-2307): the champion DoG-v2 detector keeps a **fixed percentile (92)** of
the Difference-of-Gaussians response, i.e. a fixed *fraction* of voxels regardless
of how many real cells a volume contains. On the sparse ``6bba_05b6850b`` family
(≈6362 true cells) that fixed fraction predicts ~40450 nodes, and the node-count
penalty crushes its adjusted edge Jaccard to 0.26. This screen replaces the fixed
percentile with a robust per-volume z-score threshold ``median + k·1.4826·MAD`` of
the DoG response (``DetectParams.mad_k``), which tracks each dataset's own noise
floor and so adapts the detection *count* to the signal content.

To sweep ``k`` cheaply we compute the expensive part — the DoG response and its
local maxima — **once per family/timepoint**, cache each candidate peak's response
value together with the per-volume ``median``/``MAD``, then for each ``k`` simply
re-threshold the cached candidates and re-link. Detection is otherwise identical to
``detect_centroids`` (same anisotropic sigma, same NMS footprint), so the cached
sweep reproduces what the real detector would produce with that ``mad_k``.

Writes ``experiments/sot2307/screen_adaptive.json`` and prints a per-k table. It
performs no Kaggle submission and mutates no champion state.

Usage::

    .venv/bin/python experiments/sot2307/screen_adaptive.py
    .venv/bin/python experiments/sot2307/screen_adaptive.py --ks 3 4 5 6 7 8
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from biohub_tracking.detect import DetectParams
from biohub_tracking.eval.score import _jaccard, adjusted_edge_jaccard, evaluate
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff
from biohub_tracking.link import LinkParams, link_centroids
from biohub_tracking.pipeline import _open_image_array

REPO = Path(__file__).resolve().parents[2]

# Same 4-family holdout as scripts/reanchor_oracle.py (SOT-2305): the exact Kaggle
# test set, test *.zarr image + train *.geff ground truth.
HOLDOUT = [
    {"name": "44b6_0113de3b", "prefix": "44b6", "image": "data/test/44b6_0113de3b.zarr", "geff": "data/train/44b6_0113de3b.geff"},
    {"name": "44b6_0b24845f", "prefix": "44b6", "image": "data/test/44b6_0b24845f.zarr", "geff": "data/train/44b6_0b24845f.geff"},
    {"name": "6bba_05b6850b", "prefix": "6bba", "image": "data/test/6bba_05b6850b.zarr", "geff": "data/train/6bba_05b6850b.geff"},
    {"name": "6bba_05db0fb1", "prefix": "6bba", "image": "data/test/6bba_05db0fb1.zarr", "geff": "data/train/6bba_05db0fb1.geff"},
]

# Champion DoG-v2 detection geometry (SOT-2272); only the threshold rule changes.
DOG = DetectParams(
    sigma_zyx=(1.0, 2.0, 2.0),
    background_sigma_zyx=(2.0, 6.0, 6.0),
    nms_size_zyx=(2, 5, 5),
    threshold_percentile=92.0,
)
LINK = LinkParams(max_distance=7.0, allow_division=False)


def _response(vol: np.ndarray, params: DetectParams) -> np.ndarray:
    """Reproduce detect_centroids' DoG response for one 3D volume."""
    vol = np.asarray(vol, dtype=np.float32)
    smoothed = ndi.gaussian_filter(vol, sigma=params.sigma_zyx)
    background = ndi.gaussian_filter(vol, sigma=params.background_sigma_zyx)
    return smoothed - background


def candidate_peaks(vol: np.ndarray, params: DetectParams) -> dict:
    """All NMS local maxima of the DoG response, with response value + robust stats.

    Returns the *unthresholded* candidate set for one timepoint so a single
    detection pass can be re-thresholded for many k values.
    """
    response = _response(vol, params)
    footprint = np.ones([2 * s + 1 for s in params.nms_size_zyx], dtype=bool)
    local_max = ndi.maximum_filter(response, footprint=footprint)
    peak_mask = response == local_max
    coords = np.argwhere(peak_mask).astype(np.float64)  # (N, 3) z,y,x
    values = response[peak_mask].astype(np.float64)
    median = float(np.median(response))
    mad = float(np.median(np.abs(response - median)))
    return {"coords": coords, "values": values, "median": median, "mad": mad}


def detections_for_k(cache_series: list[dict], k: float) -> dict[int, np.ndarray]:
    """Re-threshold cached candidates at mad_k=k and order brightest-first."""
    out: dict[int, np.ndarray] = {}
    for t, c in enumerate(cache_series):
        thr = c["median"] + k * 1.4826 * c["mad"]
        keep = c["values"] > thr
        coords = c["coords"][keep]
        vals = c["values"][keep]
        if coords.size:
            order = np.argsort(vals)[::-1]
            coords = coords[order]
        out[t] = coords.reshape(-1, 3)
    return out


def score_from_detections(detections, gt, scale, n_true) -> dict:
    graph = link_centroids(detections, scale=scale, params=LINK)
    r = evaluate(graph, gt, scale=scale)
    j = _jaccard(r.edge_tp, r.edge_fp, r.edge_fn)
    adj = adjusted_edge_jaccard(j, r.num_pred_nodes, n_true)
    if adj != adj:
        adj = j
    return {
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
    tp = sum(r["edge_tp"] for r in rows)
    fp = sum(r["edge_fp"] for r in rows)
    fn = sum(r["edge_fn"] for r in rows)
    edge_j = _jaccard(tp, fp, fn)
    wsum = sum(r["weight"] * r["adjusted_edge_jaccard"] for r in rows if r["weight"] > 0)
    wtot = sum(r["weight"] for r in rows if r["weight"] > 0)
    adj = wsum / wtot if wtot > 0 else float("nan")
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
    ap.add_argument("--ks", type=float, nargs="+",
                    default=[2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    ap.add_argument("--out", default="experiments/sot2307/screen_adaptive.json")
    args = ap.parse_args()

    # Load GT + build candidate cache once per family (the expensive DoG pass).
    families = []
    for fam in HOLDOUT:
        print(f"caching candidates {fam['name']} ...", flush=True)
        geff = REPO / fam["geff"]
        arr = _open_image_array(REPO / fam["image"])
        n_t = arr.shape[0]
        cache = [candidate_peaks(arr[t], DOG) for t in range(n_t)]
        families.append({
            "fam": fam,
            "gt": load_geff(geff),
            "scale": geff_scale(geff),
            "n_true": geff_estimated_num_nodes(geff),
            "cache": cache,
        })
        tot = sum(len(c["values"]) for c in cache)
        print(f"  {fam['name']}: {n_t} timepoints, {tot} candidate peaks", flush=True)

    per_k = {}
    for k in args.ks:
        rows = []
        for f in families:
            dets = detections_for_k(f["cache"], k)
            row = score_from_detections(dets, f["gt"], f["scale"], f["n_true"])
            row["name"] = f["fam"]["name"]
            row["prefix"] = f["fam"]["prefix"]
            rows.append(row)
        overall = micro(rows)
        by_prefix = {
            p: micro([r for r in rows if r["prefix"] == p])
            for p in sorted({r["prefix"] for r in rows})
        }
        per_k[f"k={k}"] = {
            "mad_k": k,
            "per_dataset": rows,
            "by_prefix": {p: v["score"] for p, v in by_prefix.items()},
            "holdout_micro": overall,
        }
        print(
            f"\nmad_k={k}: holdout micro-adj={overall['micro_adj_edge_jaccard']} "
            f"by_prefix={{ {', '.join(f'{p}:{v}' for p, v in per_k[f'k={k}']['by_prefix'].items())} }}",
            flush=True,
        )
        for r in rows:
            print(
                f"    {r['name']:16s} tp/fp/fn={r['edge_tp']}/{r['edge_fp']}/{r['edge_fn']} "
                f"adj={r['adjusted_edge_jaccard']} pred_nodes={r['pred_nodes']} n_true={r['n_true']}",
                flush=True,
            )

    result = {
        "issue": "SOT-2307",
        "title": "Screen adaptive robust (median+k*MAD) DoG threshold on the 4-dataset holdout",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_champion_dog_v2_percentile92": {
            "holdout_micro_adj": 0.5225,
            "by_prefix": {"44b6": 0.7721, "6bba": 0.5117},
            "note": "docs/reanchored-oracle-evaluation.json (SOT-2305)",
        },
        "holdout": [f["name"] for f in HOLDOUT],
        "detector": "DoG-v2 geometry, threshold = median + mad_k*1.4826*MAD per volume",
        "sweep": per_k,
        "reproducible_note": "Deterministic (no RNG); candidate cache reproduces detect_centroids.",
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
