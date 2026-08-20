"""Screen the non-destructive division overlay (SOT-2818) on the re-anchored CV.

The champion ``detect-link-dog-v4-shorttrack`` runs ``allow_division=False`` and
scores division Jaccard 0.0. This screens the post-processing division overlay
(``biohub_tracking.division_overlay``) — which *adds* second-daughter edges to the
final champion graph without touching the linking assignment — against the frozen
champion baseline over a high-precision gate grid:

* ``max_distance``      ∈ {7, 5, 3} µm     — parent→2nd-daughter radius
* ``sibling_ratio``     ∈ {0(off),2,1.5,1} — d2 <= ratio·d1 sibling-balance gate
* ``min_daughter_len``  ∈ {2, 3, 4}        — 2nd-daughter track persistence

Detection is the frozen champion DoG-v3 adaptive detector, computed **once** per
family and cached; every overlay variant is re-linked (champion link + overlay)
off the cached detections and scored through the one SOT-2761/2817 re-anchored CV
aggregation (``eval.cv.aggregate``), so the numbers are byte-comparable to the
registry champion CV (0.6649) and the reported ``score`` is the *full* competition
metric ``adjusted_edge_jaccard + 0.1·division_jaccard`` (SOT-2817).

Promotion gate: combined ``score`` strictly above the champion baseline **and**
per-family edge non-regression (``CvResult.no_regression_vs`` against the live
champion per-dataset adjusted edge Jaccard). Writes
``experiments/sot2818-division-overlay/screen_overlay.json``. No Kaggle submission.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.detect import detect_volume_series
from biohub_tracking.eval.cv import (
    CHAMPION_REFERENCE_MICRO_ADJ,
    CV_HOLDOUT,
    aggregate,
    cv_result_to_dict,
    score_family,
)
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff
from biohub_tracking.link import LinkParams, link_centroids
from biohub_tracking.pipeline import _open_image_array

REPO = Path(__file__).resolve().parents[2]

MAX_DISTANCES = [7.0, 5.0, 3.0]
SIBLING_RATIOS = [0.0, 2.0, 1.5, 1.0]
MIN_DAUGHTER_LENS = [2, 3, 4]
REQUIRE_PARENT_TRACK = True


def cv_for_link(cache: dict, link: LinkParams):
    rows = []
    for fam in CV_HOLDOUT:
        dets, gt, scale, n_true = cache[fam.name]
        pred = link_centroids(dets, scale=scale, params=link)
        rows.append(score_family(fam, pred, gt, n_true, scale=scale))
    return aggregate(rows)


def main() -> int:
    cfg = load_champion_config()
    detect, base_link, _cfg_scale = champion_params(cfg)

    cache: dict = {}
    for fam in CV_HOLDOUT:
        t0 = time.time()
        arr = _open_image_array(REPO / fam.image)
        dets = detect_volume_series(arr, detect)
        gt = load_geff(REPO / fam.geff)
        scale = geff_scale(REPO / fam.geff)
        n_true = geff_estimated_num_nodes(REPO / fam.geff)
        cache[fam.name] = (dets, gt, scale, n_true)
        print(f"[detect] {fam.name}: {time.time()-t0:.1f}s "
              f"dets={sum(len(v) for v in dets.values())}", flush=True)

    baseline = cv_for_link(cache, base_link)
    floor = {r.name: r.adj_edge_jaccard for r in baseline.per_dataset}
    print(f"[baseline] champion micro_adj={baseline.micro_adj_edge_jaccard:.4f} "
          f"div_j={baseline.division_jaccard} score={baseline.score:.4f}", flush=True)
    print(f"[baseline] per-dataset adj floor: "
          f"{ {k: round(v,4) for k,v in floor.items()} }", flush=True)

    variants = []
    for md in MAX_DISTANCES:
        for ratio in SIBLING_RATIOS:
            for mdl in MIN_DAUGHTER_LENS:
                overlay = ("nearest-head", md, ratio, mdl, REQUIRE_PARENT_TRACK)
                link = replace(base_link, division_overlay=overlay)
                res = cv_for_link(cache, link)
                per = {r.name: r.adj_edge_jaccard for r in res.per_dataset}
                no_reg = res.no_regression_vs(floor)
                score_up = res.score > baseline.score + 1e-9
                div = (None if res.division_jaccard != res.division_jaccard
                       else round(res.division_jaccard, 4))
                variants.append({
                    "max_distance": md,
                    "sibling_ratio": ratio,
                    "min_daughter_len": mdl,
                    "require_parent_track": REQUIRE_PARENT_TRACK,
                    "micro_adj_edge_jaccard": round(res.micro_adj_edge_jaccard, 4),
                    "division_jaccard": div,
                    "score": round(res.score, 4),
                    "delta_score_vs_champion": round(res.score - baseline.score, 4),
                    "no_per_dataset_regression": bool(no_reg),
                    "promotable": bool(no_reg and score_up),
                    "per_dataset_adj": {k: round(v, 4) for k, v in per.items()},
                    "division_by_dataset": {
                        r.name: {"tp": r.division_tp, "fp": r.division_fp,
                                 "fn": r.division_fn}
                        for r in res.per_dataset
                    },
                    "edge_by_dataset": {
                        r.name: {"tp": r.edge_tp, "fp": r.edge_fp, "fn": r.edge_fn}
                        for r in res.per_dataset
                    },
                    "cv": cv_result_to_dict(res),
                })
                print(f"[variant] md={md} ratio={ratio} mdl={mdl} "
                      f"micro_adj={res.micro_adj_edge_jaccard:.4f} div_j={div} "
                      f"score={res.score:.4f} "
                      f"d={variants[-1]['delta_score_vs_champion']:+.4f} "
                      f"no_reg={no_reg} promotable={variants[-1]['promotable']}",
                      flush=True)

    variants.sort(key=lambda v: v["score"], reverse=True)
    promotable = [v for v in variants if v["promotable"]]

    # A variant that fires at all (division confusion matrix non-empty on some
    # family) — the ablation-firing evidence the acceptance criteria require.
    fired = [
        v for v in variants
        if any(d["tp"] + d["fp"] + d["fn"] > 0
               for d in v["division_by_dataset"].values())
    ]

    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2818",
        "axis": "non-destructive division-event overlay (post-link second-daughter "
                "edge addition; linking assignment unchanged) vs the frozen "
                "detect-link-dog-v4-shorttrack champion; single-variable same-seed "
                "ablation on the SOT-2817 re-anchored full-metric CV",
        "cv_source": "biohub_tracking.eval.cv (SOT-2761 leak-free 4-family holdout, "
                     "SOT-2817 full-metric re-anchor)",
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "baseline_champion": {
            "micro_adj_edge_jaccard": round(baseline.micro_adj_edge_jaccard, 4),
            "division_jaccard": (None if baseline.division_jaccard != baseline.division_jaccard
                                 else round(baseline.division_jaccard, 4)),
            "score": round(baseline.score, 4),
            "cv": cv_result_to_dict(baseline),
        },
        "champion_per_dataset_adj_floor": {k: round(v, 4) for k, v in floor.items()},
        "grid": {
            "max_distance": MAX_DISTANCES,
            "sibling_ratio": SIBLING_RATIOS,
            "min_daughter_len": MIN_DAUGHTER_LENS,
            "require_parent_track": REQUIRE_PARENT_TRACK,
        },
        "variants_ranked_by_score": variants,
        "n_fired": len(fired),
        "n_promotable": len(promotable),
        "best_promotable": promotable[0] if promotable else None,
    }
    out = REPO / "experiments/sot2818-division-overlay/screen_overlay.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"baseline champion score={baseline.score:.4f} "
          f"(micro_adj={baseline.micro_adj_edge_jaccard:.4f}, "
          f"div_j={baseline.division_jaccard})")
    print(f"n_fired={len(fired)} n_promotable={len(promotable)}")
    if promotable:
        b = promotable[0]
        print(f"BEST PROMOTABLE: md={b['max_distance']} ratio={b['sibling_ratio']} "
              f"mdl={b['min_daughter_len']} score={b['score']} "
              f"(+{b['delta_score_vs_champion']}) div_j={b['division_jaccard']}")
    else:
        top = variants[0]
        print(f"NO PROMOTABLE variant. Top-by-score: md={top['max_distance']} "
              f"ratio={top['sibling_ratio']} mdl={top['min_daughter_len']} "
              f"score={top['score']} (delta {top['delta_score_vs_champion']:+}) "
              f"no_reg={top['no_per_dataset_regression']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
