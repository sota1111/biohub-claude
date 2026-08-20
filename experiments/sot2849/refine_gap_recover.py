"""Refine node-interpolation gap recovery min_frag (SOT-2849 confirm step).

The screen (screen_gap_recover.json) showed the mechanism recovers real FN edges
(micro-adj 0.6649 -> up to 0.6773, dTP>0/dFN<0) but every variant failed the
per-dataset no-regression gate on ONE family: the ultra-sparse 44b6_0113de3b,
whose edge delta was (d_tp=0, d_fp=0, d_fn=0) — a *pure node-count penalty* from
interpolated nodes that recover no scored edge on that sparse family. Raising
min_frag toward the champion min_track_length=4 should refuse to bridge the short
noise fragments that generate those non-recovering interpolations, killing the
44b6 node-count churn while keeping the dense-family FN recovery.

Same cached-detection, single-variable, same-seed A/B as the screen; only
min_frag (and mg/dist) vary. Writes experiments/sot2849/refine_gap_recover.json.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.detect import detect_volume_series
from biohub_tracking.eval.cv import (
    CHAMPION_REFERENCE_MICRO_ADJ,
    CV_HOLDOUT,
    cv_result_to_dict,
)
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff
from biohub_tracking.pipeline import _open_image_array

from experiments.sot2849.screen_gap_recover import (
    CHAMPION_PER_DATASET_ADJ,
    cv_for_link,
    edge_counts_by_dataset,
    variant_link,
)

REPO = Path(__file__).resolve().parents[2]

MAX_GAPS = [2, 3]
DISTANCES = [7.0, 10.0]
MIN_FRAGS = [4, 5, 6, 8]


def main() -> int:
    cfg = load_champion_config()
    detect, base_link, _s = champion_params(cfg)

    cache: dict = {}
    for fam in CV_HOLDOUT:
        t0 = time.time()
        arr = _open_image_array(REPO / fam.image)
        dets = detect_volume_series(arr, detect)
        gt = load_geff(REPO / fam.geff)
        scale = geff_scale(REPO / fam.geff)
        n_true = geff_estimated_num_nodes(REPO / fam.geff)
        cache[fam.name] = (dets, gt, scale, n_true)
        print(f"[detect] {fam.name}: {time.time()-t0:.1f}s", flush=True)

    baseline = cv_for_link(cache, base_link)
    base_ec = edge_counts_by_dataset(baseline)
    print(f"[baseline] champion micro_adj={baseline.micro_adj_edge_jaccard:.4f}", flush=True)

    variants = []
    for mg in MAX_GAPS:
        for dist in DISTANCES:
            for mf in MIN_FRAGS:
                link = variant_link(base_link, mg, dist, mf)
                res = cv_for_link(cache, link)
                per = {r.name: r.adj_edge_jaccard for r in res.per_dataset}
                no_reg = res.no_regression_vs(CHAMPION_PER_DATASET_ADJ)
                score_up = res.score > baseline.score + 1e-9
                ec = edge_counts_by_dataset(res)
                edge_delta = {
                    name: {
                        "d_tp": ec[name]["tp"] - base_ec[name]["tp"],
                        "d_fp": ec[name]["fp"] - base_ec[name]["fp"],
                        "d_fn": ec[name]["fn"] - base_ec[name]["fn"],
                        "d_pred_nodes": ec[name]["pred_nodes"] - base_ec[name]["pred_nodes"],
                    }
                    for name in ec
                }
                tot = {k: sum(v[k] for v in edge_delta.values())
                       for k in ("d_tp", "d_fp", "d_fn")}
                variants.append({
                    "gap_recover_max_gap": mg,
                    "gap_recover_distance": dist,
                    "gap_recover_min_frag": mf,
                    "micro_adj_edge_jaccard": round(res.micro_adj_edge_jaccard, 4),
                    "score": round(res.score, 4),
                    "delta_score_vs_champion": round(res.score - baseline.score, 4),
                    "no_per_dataset_regression": bool(no_reg),
                    "promotable": bool(no_reg and score_up),
                    "total_edge_delta": tot,
                    "per_dataset_adj": {k: round(v, 4) for k, v in per.items()},
                    "edge_delta_by_dataset": edge_delta,
                    "cv": cv_result_to_dict(res),
                })
                print(f"[variant] mg={mg} dist={dist} mf={mf} "
                      f"micro_adj={res.micro_adj_edge_jaccard:.4f} "
                      f"d={variants[-1]['delta_score_vs_champion']:+.4f} "
                      f"dTP={tot['d_tp']:+d} dFP={tot['d_fp']:+d} dFN={tot['d_fn']:+d} "
                      f"no_reg={no_reg} promotable={variants[-1]['promotable']}", flush=True)

    variants.sort(key=lambda v: (v["promotable"], v["score"]), reverse=True)
    promotable = [v for v in variants if v["promotable"]]

    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2849",
        "axis": "node-interpolation gap recovery min_frag refine (confirm step): raise the "
                "per-terminal fragment-size gate toward min_track_length=4 to drop pure "
                "node-count interpolations on the sparse 44b6_0113de3b family",
        "cv_source": "biohub_tracking.eval.cv (SOT-2761 leak-free 4-family holdout)",
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "baseline_champion": {
            "micro_adj_edge_jaccard": round(baseline.micro_adj_edge_jaccard, 4),
            "score": round(baseline.score, 4),
            "edge_counts_by_dataset": base_ec,
        },
        "champion_per_dataset_adj_floor": CHAMPION_PER_DATASET_ADJ,
        "grid": {"gap_recover_max_gap": MAX_GAPS, "gap_recover_distance": DISTANCES,
                 "gap_recover_min_frag": MIN_FRAGS},
        "variants_ranked": variants,
        "n_promotable": len(promotable),
        "best_promotable": promotable[0] if promotable else None,
    }
    out = REPO / "experiments/sot2849/refine_gap_recover.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}\nn_promotable={len(promotable)}")
    if promotable:
        b = promotable[0]
        print(f"BEST PROMOTABLE: mg={b['gap_recover_max_gap']} dist={b['gap_recover_distance']} "
              f"mf={b['gap_recover_min_frag']} micro={b['micro_adj_edge_jaccard']} "
              f"(+{b['delta_score_vs_champion']}) per={b['per_dataset_adj']} "
              f"edge_delta={b['total_edge_delta']}")
    else:
        top = variants[0]
        print(f"NO PROMOTABLE. Top: mg={top['gap_recover_max_gap']} dist={top['gap_recover_distance']} "
              f"mf={top['gap_recover_min_frag']} micro={top['micro_adj_edge_jaccard']} "
              f"(d {top['delta_score_vs_champion']:+}) no_reg={top['no_per_dataset_regression']} "
              f"per={top['per_dataset_adj']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
