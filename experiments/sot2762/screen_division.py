"""Screen division-aware linking (SOT-2762) on the SOT-2761 leak-free CV.

The champion ``detect-link-dog-v4-shorttrack`` runs ``link.allow_division=False``
and therefore scores division Jaccard 0.0 — the 6bba_05db0fb1 family has 3 GT
divisions, all missed. Zebrafish embryo development is division-heavy, so
division-aware linking is an untapped score lever (the official metric is
``adjusted_edge_jaccard + 0.1 * division_jaccard``).

This screens ``allow_division=True`` against the frozen champion baseline
(single-variable ablation; the pipeline is deterministic, so this is a same-seed
A/B) over two over-split suppressors:

* ``division_distance`` in {7,5,3,2} µm — tighter = fewer, higher-confidence forks
* ``division_max_sibling_ratio`` in {0(off),3,2,1.5} — sibling-balance gate

Detection is frozen (champion DoG-v3 adaptive ``mad_k=3.0``) and computed **once**
per family; every link variant is re-linked off the cached detections and scored
through the one SOT-2761 CV aggregation (``eval.cv.score_family`` / ``aggregate``),
so the numbers are byte-comparable to the registry champion CV (0.6649).

Writes ``experiments/sot2762/screen_division.json``. No Kaggle submission.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.eval.cv import (
    CHAMPION_REFERENCE_MICRO_ADJ,
    CV_HOLDOUT,
    aggregate,
    cv_result_to_dict,
    score_family,
)
from biohub_tracking.detect import detect_volume_series
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff
from biohub_tracking.link import LinkParams, link_centroids
from biohub_tracking.pipeline import _open_image_array

REPO = Path(__file__).resolve().parents[2]

# Champion per-dataset adjusted edge Jaccard — the no-regression floor (registry).
CHAMPION_PER_DATASET_ADJ = {
    "44b6_0113de3b": 0.8895,
    "44b6_0b24845f": 0.6817,
    "6bba_05b6850b": 0.5700,
    "6bba_05db0fb1": 0.7310,
}

DIVISION_DISTANCES = [7.0, 5.0, 3.0, 2.0]
SIBLING_RATIOS = [0.0, 3.0, 2.0, 1.5]


def champion_link(cfg: dict) -> LinkParams:
    _detect, link, _scale = champion_params(cfg)
    return link


def variant_link(base: LinkParams, division_distance: float, ratio: float) -> LinkParams:
    """Champion link params but with division enabled + over-split knobs set."""
    return LinkParams(
        max_distance=base.max_distance,
        allow_division=True,
        division_distance=division_distance,
        division_max_sibling_ratio=ratio,
        velocity_gain=base.velocity_gain,
        velocity_disp_weight=base.velocity_disp_weight,
        motion_gate_on_prediction=base.motion_gate_on_prediction,
        min_track_length=base.min_track_length,
    )


def cv_for_link(cache: dict, link: LinkParams):
    """Aggregate the leak-free CV for a given LinkParams off cached detections."""
    rows = []
    for fam in CV_HOLDOUT:
        dets, gt, scale, n_true = cache[fam.name]
        pred = link_centroids(dets, scale=scale, params=link)
        rows.append(score_family(fam, pred, gt, n_true, scale=scale))
    return aggregate(rows)


def main() -> int:
    cfg = load_champion_config()
    detect, base_link, _cfg_scale = champion_params(cfg)

    # Detect once per family; cache (detections, gt, scale, n_true).
    cache: dict = {}
    for fam in CV_HOLDOUT:
        t0 = time.time()
        image = REPO / fam.image
        geff = REPO / fam.geff
        arr = _open_image_array(image)
        dets = detect_volume_series(arr, detect)
        gt = load_geff(geff)
        scale = geff_scale(geff)
        n_true = geff_estimated_num_nodes(geff)
        cache[fam.name] = (dets, gt, scale, n_true)
        print(f"[detect] {fam.name}: {time.time()-t0:.1f}s "
              f"dets={sum(len(v) for v in dets.values())}", flush=True)

    # Baseline: the frozen champion link (allow_division=False).
    baseline = cv_for_link(cache, base_link)
    print(f"[baseline] champion micro_adj={baseline.micro_adj_edge_jaccard:.4f} "
          f"div_j={baseline.division_jaccard} score={baseline.score:.4f}", flush=True)

    variants = []
    for dd in DIVISION_DISTANCES:
        for ratio in SIBLING_RATIOS:
            link = variant_link(base_link, dd, ratio)
            res = cv_for_link(cache, link)
            per = {r.name: r.adj_edge_jaccard for r in res.per_dataset}
            no_reg = res.no_regression_vs(CHAMPION_PER_DATASET_ADJ)
            score_up = res.score > baseline.score + 1e-9
            variants.append({
                "division_distance": dd,
                "division_max_sibling_ratio": ratio,
                "micro_adj_edge_jaccard": round(res.micro_adj_edge_jaccard, 4),
                "division_jaccard": (None if res.division_jaccard != res.division_jaccard
                                     else round(res.division_jaccard, 4)),
                "score": round(res.score, 4),
                "delta_score_vs_champion": round(res.score - baseline.score, 4),
                "no_per_dataset_regression": bool(no_reg),
                "promotable": bool(no_reg and score_up),
                "per_dataset_adj": {k: round(v, 4) for k, v in per.items()},
                "division_by_dataset": {
                    r.name: {"tp": r.division_tp, "fp": r.division_fp, "fn": r.division_fn}
                    for r in res.per_dataset
                },
                "cv": cv_result_to_dict(res),
            })
            print(f"[variant] dd={dd} ratio={ratio} "
                  f"micro_adj={res.micro_adj_edge_jaccard:.4f} "
                  f"div_j={variants[-1]['division_jaccard']} "
                  f"score={res.score:.4f} d={variants[-1]['delta_score_vs_champion']:+.4f} "
                  f"no_reg={no_reg} promotable={variants[-1]['promotable']}", flush=True)

    variants.sort(key=lambda v: v["score"], reverse=True)
    promotable = [v for v in variants if v["promotable"]]

    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2762",
        "axis": "division-aware linking (allow_division on) + over-split control "
                "(division_distance, division_max_sibling_ratio), single-variable "
                "ablation vs the frozen detect-link-dog-v4-shorttrack champion",
        "cv_source": "biohub_tracking.eval.cv (SOT-2761 leak-free 4-family holdout)",
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "baseline_champion": {
            "micro_adj_edge_jaccard": round(baseline.micro_adj_edge_jaccard, 4),
            "division_jaccard": (None if baseline.division_jaccard != baseline.division_jaccard
                                 else round(baseline.division_jaccard, 4)),
            "score": round(baseline.score, 4),
            "cv": cv_result_to_dict(baseline),
        },
        "champion_per_dataset_adj_floor": CHAMPION_PER_DATASET_ADJ,
        "grid": {"division_distance": DIVISION_DISTANCES, "division_max_sibling_ratio": SIBLING_RATIOS},
        "variants_ranked_by_score": variants,
        "n_promotable": len(promotable),
        "best_promotable": promotable[0] if promotable else None,
    }
    out = REPO / "experiments/sot2762/screen_division.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"baseline champion score={baseline.score:.4f} "
          f"(micro_adj={baseline.micro_adj_edge_jaccard:.4f}, div_j={baseline.division_jaccard})")
    print(f"n_promotable={len(promotable)}")
    if promotable:
        b = promotable[0]
        print(f"BEST PROMOTABLE: dd={b['division_distance']} ratio={b['division_max_sibling_ratio']} "
              f"score={b['score']} (+{b['delta_score_vs_champion']}) div_j={b['division_jaccard']}")
    else:
        top = variants[0]
        print(f"NO PROMOTABLE variant. Top-by-score: dd={top['division_distance']} "
              f"ratio={top['division_max_sibling_ratio']} score={top['score']} "
              f"(delta {top['delta_score_vs_champion']:+}) no_reg={top['no_per_dataset_regression']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
