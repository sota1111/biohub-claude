"""Screen recall-oriented FN-edge-endpoint recovery (SOT-2873) on the leak-free CV.

Objective = **GT-node recall @7 µm** (not the aggregate score, and not a
family-invariant per-voxel operating point — the axis SOT-2789 / SOT-2848,2863
exhausted). The competition edge metric scores an edge TP only when both endpoints
match a GT node within 7 µm but charges **no node FP** for an unmatched predicted
node (sparse GT); it only applies the mild *global* over-prediction penalty
``J·(1 − 0.1·(N_pred − N_true)/N_true)``. So admitting sub-threshold detections to
recover missed FN-edge endpoints is a one-directional gain *unless* the
over-prediction penalty cancels it. This screen measures that tradeoff directly.

Method (single-variable same-seed A/B vs the frozen champion). Detection is the
champion DoG-v3 adaptive detector; each variant re-runs the REAL
``detect.recall_recovery`` code path (``("madk_tier", k_low, max_extra_frac)``),
re-links with the champion linker, and is scored through the one SOT-2761 CV
aggregation (byte-comparable to the registry champion 0.6649). For every variant
we record, per family and micro: GT-node recall @7 µm (overall + edge-endpoint),
the adjusted edge Jaccard, edge TP/FP/FN deltas, predicted-node count, and the
implied over-prediction penalty factor — i.e. the recall-vs-penalty tradeoff curve.

Promotion gate (mandatory): per-dataset no-regression AND micro score up. A null
result is a clean evidence-backed reject/inconclusive, not a false promotion.
No champion mutation, no Kaggle submission. Writes screen_recall.json.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.detect import DetectParams, detect_volume_series
from biohub_tracking.eval.cv import (
    CHAMPION_REFERENCE_MICRO_ADJ,
    CV_HOLDOUT,
    aggregate,
    cv_result_to_dict,
    score_family,
)
from biohub_tracking.eval.recall_metric import gt_node_recall
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff
from biohub_tracking.link import link_centroids
from biohub_tracking.pipeline import _open_image_array

REPO = Path(__file__).resolve().parents[2]

# Champion per-dataset adjusted edge Jaccard — the no-regression floor (registry).
CHAMPION_PER_DATASET_ADJ = {
    "44b6_0113de3b": 0.8895,
    "44b6_0b24845f": 0.6817,
    "6bba_05b6850b": 0.5700,
    "6bba_05db0fb1": 0.7310,
}

# Recall-vs-penalty tradeoff grid. k_low < mad_k(=3.0) sets how far below the
# champion cutoff the recovery band reaches; max_extra_frac caps the added tier as
# a fraction of the champion primary count (the over-prediction-penalty control).
K_LOWS = [2.0, 1.0]
MAX_EXTRA_FRACS = [0.1, 0.3, 0.6]


def _detect_params_with_recall(base: DetectParams, k_low: float, frac: float) -> DetectParams:
    return DetectParams(**{**base.__dict__, "recall_recovery": ("madk_tier", k_low, frac)})


def _penalty_factor(n_pred: int, n_true: float) -> float:
    """The over-prediction penalty multiplier applied to the edge Jaccard."""
    if n_true is None or n_true <= 0:
        return float("nan")
    return max(0.0, 1.0 - 0.1 * (n_pred - n_true) / n_true)


def _recall_for_dets(dets, gt, scale, n_true, link, fam_name):
    """Score one family's detections: CV row + GT-node recall @7 µm + penalty."""
    pred = link_centroids(dets, scale=scale, params=link)
    row = score_family(_fam_by_name(fam_name), pred, gt, n_true, scale=scale)
    rec = gt_node_recall(pred, gt, scale=scale, max_distance=7.0)
    return row, rec, pred.num_nodes


def _fam_by_name(name):
    for fam in CV_HOLDOUT:
        if fam.name == name:
            return fam
    raise KeyError(name)


def _recall_summary(recs: dict) -> dict:
    """Micro (pooled) + per-family GT-node recall @7 µm."""
    tot_ep = sum(r.gt_edge_endpoint_nodes for r in recs.values())
    tot_ep_m = sum(r.gt_edge_endpoint_matched for r in recs.values())
    tot_n = sum(r.gt_nodes for r in recs.values())
    tot_n_m = sum(r.gt_nodes_matched for r in recs.values())
    return {
        "micro_gt_node_recall": round(tot_n_m / tot_n, 4) if tot_n else None,
        "micro_gt_edge_endpoint_recall": round(tot_ep_m / tot_ep, 4) if tot_ep else None,
        "per_family": {
            name: {
                "gt_node_recall": round(r.gt_node_recall, 4),
                "gt_edge_endpoint_recall": round(r.gt_edge_endpoint_recall, 4),
                "gt_edge_endpoint_matched": r.gt_edge_endpoint_matched,
                "gt_edge_endpoint_nodes": r.gt_edge_endpoint_nodes,
            }
            for name, r in recs.items()
        },
    }


def main() -> int:
    cfg = load_champion_config()
    base_detect, base_link, _scale = champion_params(cfg)

    # Cache GT/scale/n_true per family once. Detection is re-run per variant (the
    # recall tier changes the detected set), which is the faithful real-code A/B.
    meta_cache: dict = {}
    for fam in CV_HOLDOUT:
        geff = REPO / fam.geff
        meta_cache[fam.name] = (
            load_geff(geff),
            geff_scale(geff),
            geff_estimated_num_nodes(geff),
        )

    def run_config(detect_params, tag: str):
        rows, recs, npred = [], {}, {}
        for fam in CV_HOLDOUT:
            gt, scale, n_true = meta_cache[fam.name]
            t0 = time.time()
            arr = _open_image_array(REPO / fam.image)
            dets = detect_volume_series(arr, detect_params)
            row, rec, n = _recall_for_dets(dets, gt, scale, n_true, base_link, fam.name)
            rows.append(row)
            recs[fam.name] = rec
            npred[fam.name] = n
            print(
                f"[{tag}] {fam.name}: {time.time()-t0:.1f}s "
                f"adj={row.adj_edge_jaccard:.4f} tp={row.edge_tp} fp={row.edge_fp} "
                f"fn={row.edge_fn} npred={n} ep_recall={rec.gt_edge_endpoint_recall:.4f}",
                flush=True,
            )
        return aggregate(rows), recs, npred

    baseline, base_recs, base_npred = run_config(base_detect, "baseline")
    base_ec = {r.name: (r.edge_tp, r.edge_fp, r.edge_fn) for r in baseline.per_dataset}
    print(
        f"[baseline] micro_adj={baseline.micro_adj_edge_jaccard:.4f} "
        f"score={baseline.score:.4f} "
        f"ep_recall={_recall_summary(base_recs)['micro_gt_edge_endpoint_recall']}",
        flush=True,
    )

    variants = []
    for k_low in K_LOWS:
        for frac in MAX_EXTRA_FRACS:
            dp = _detect_params_with_recall(base_detect, k_low, frac)
            res, recs, npred = run_config(dp, f"k{k_low}_f{frac}")
            per = {r.name: r.adj_edge_jaccard for r in res.per_dataset}
            no_reg = res.no_regression_vs(CHAMPION_PER_DATASET_ADJ)
            score_up = res.score > baseline.score + 1e-9
            ec = {r.name: (r.edge_tp, r.edge_fp, r.edge_fn) for r in res.per_dataset}
            edge_delta = {
                name: {
                    "d_tp": ec[name][0] - base_ec[name][0],
                    "d_fp": ec[name][1] - base_ec[name][1],
                    "d_fn": ec[name][2] - base_ec[name][2],
                    "d_npred": npred[name] - base_npred[name],
                }
                for name in ec
            }
            tot = {
                k: sum(v[k] for v in edge_delta.values())
                for k in ("d_tp", "d_fp", "d_fn", "d_npred")
            }
            rec_sum = _recall_summary(recs)
            penalties = {
                r.name: round(_penalty_factor(r.num_pred_nodes, r.n_true), 4)
                for r in res.per_dataset
            }
            variants.append({
                "k_low": k_low,
                "max_extra_frac": frac,
                "micro_adj_edge_jaccard": round(res.micro_adj_edge_jaccard, 4),
                "score": round(res.score, 4),
                "delta_score_vs_champion": round(res.score - baseline.score, 4),
                "no_per_dataset_regression": bool(no_reg),
                "promotable": bool(no_reg and score_up),
                "recall": rec_sum,
                "delta_micro_ep_recall": (
                    round(
                        rec_sum["micro_gt_edge_endpoint_recall"]
                        - _recall_summary(base_recs)["micro_gt_edge_endpoint_recall"],
                        4,
                    )
                    if rec_sum["micro_gt_edge_endpoint_recall"] is not None
                    else None
                ),
                "total_edge_delta": tot,
                "per_dataset_adj": {k: round(v, 4) for k, v in per.items()},
                "edge_delta_by_dataset": edge_delta,
                "over_prediction_penalty_factor": penalties,
                "cv": cv_result_to_dict(res),
            })
            v = variants[-1]
            print(
                f"[variant] k_low={k_low} frac={frac} "
                f"micro_adj={res.micro_adj_edge_jaccard:.4f} score={res.score:.4f} "
                f"d={v['delta_score_vs_champion']:+.4f} "
                f"ep_recall={rec_sum['micro_gt_edge_endpoint_recall']} "
                f"(d={v['delta_micro_ep_recall']:+}) "
                f"dTP={tot['d_tp']:+d} dFP={tot['d_fp']:+d} dFN={tot['d_fn']:+d} "
                f"dNpred={tot['d_npred']:+d} no_reg={no_reg} promotable={v['promotable']}",
                flush=True,
            )

    variants.sort(key=lambda v: v["score"], reverse=True)
    promotable = [v for v in variants if v["promotable"]]

    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2873",
        "axis": "recall-oriented FN-edge-endpoint recovery (detect.recall_recovery "
                "madk_tier: bounded strongest-first sub-threshold local-maxima tier), "
                "objective=GT-node recall@7um, single-variable same-seed A/B vs the "
                "frozen detect-link-dog-v4-shorttrack champion",
        "cv_source": "biohub_tracking.eval.cv (SOT-2761 leak-free 4-family holdout)",
        "objective": "GT-node recall @7um (edge-endpoint), measured against the "
                     "over-prediction-penalty tradeoff, under the metric's no-node-FP "
                     "property for unmatched predictions.",
        "distinct_from_sot2789": "SOT-2789/2848/2863 searched for a family-invariant "
                                 "per-voxel operating point on the aggregate score and "
                                 "found none. This does NOT move the champion operating "
                                 "point (primary tier byte-frozen) and does not claim a "
                                 "global magnitude; it adds a capped recall tier and "
                                 "reports the recall-vs-penalty tradeoff explicitly.",
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "champion_per_dataset_adj_floor": CHAMPION_PER_DATASET_ADJ,
        "grid": {"k_low": K_LOWS, "max_extra_frac": MAX_EXTRA_FRACS},
        "baseline_champion": {
            "micro_adj_edge_jaccard": round(baseline.micro_adj_edge_jaccard, 4),
            "score": round(baseline.score, 4),
            "recall": _recall_summary(base_recs),
            "edge_counts_by_dataset": {k: list(v) for k, v in base_ec.items()},
            "pred_nodes_by_dataset": base_npred,
            "cv": cv_result_to_dict(baseline),
        },
        "variants_ranked_by_score": variants,
        "n_promotable": len(promotable),
        "best_promotable": promotable[0] if promotable else None,
    }
    out = REPO / "experiments/sot2866c/screen_recall.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"baseline champion score={baseline.score:.4f}; n_promotable={len(promotable)}")
    if not promotable:
        top = variants[0]
        print(
            f"NO PROMOTABLE variant. Top-by-score: k_low={top['k_low']} "
            f"frac={top['max_extra_frac']} score={top['score']} "
            f"(delta {top['delta_score_vs_champion']:+}) "
            f"ep_recall_delta={top['delta_micro_ep_recall']:+} "
            f"no_reg={top['no_per_dataset_regression']} edge_delta={top['total_edge_delta']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
