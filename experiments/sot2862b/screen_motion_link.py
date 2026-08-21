"""Screen ARGUS motion-model predicted-position LAP linking (SOT-2864) on the
SOT-2761 leak-free CV.

Axis (issue SOT-2864, cycle-9 child B). champion's linker matches ``t -> t+1`` by
raw scaled distance (memoryless nearest-neighbour). ARGUS (arXiv:2607.08297, CTC
0.90-0.97, CPU-only / no weights) instead matches against *where each cell is
predicted to move* under a motion model. We port that core idea as a **global,
spatially-smoothed motion field** estimated from the current frame pair
(``motion_model_link``): a provisional assignment gives anchor displacements, a
Gaussian kernel (``motion_smooth_sigma`` scaled microns) diffuses them into a dense
field, and the final LAP is solved on the predicted source positions.

Portability (issue acceptance): ``cv2`` (Farneback dense flow) is NOT installed in
this repo/offline kernel, so the numpy/scipy-only detection-cloud field is the
first-class path — pure ``numpy``/``scipy``, no weights/internet — which the
exec-compat gate confirms importable. When ``cv2`` is present it could replace the
field estimator, but the fallback carries the mechanism portably.

Distinct from prior linking axes:
* constant-velocity ``velocity_gain`` (SOT-2369) predicts each cell from *its own*
  ``t-1 -> t`` edge, so a first-appearance cell gets no prediction; the motion
  field predicts *every* source (history-less included) from its neighbourhood.
* static gap-closing (SOT-2763, rejected) / node-interp gap-recovery (SOT-2849,
  rejected) touch missing/gap frames; this changes the primary ``t -> t+1`` cost.

Measured on the **champion classical detection** (learned detection SOT-2848 is
degenerate, micro-adj 0.0, no family-robust operating point — a linking A/B is only
meaningful on the champion detector). Detection is computed once per family and
cached; every link variant is re-linked off the cache and scored through the one
SOT-2761 CV aggregation, so numbers are byte-comparable to the champion CV (0.6649)
== a same-seed A/B (the whole pipeline is deterministic).

Grid:
* motion_smooth_sigma in {8, 15, 30} um  (local field -> near-rigid global field)
* motion_gain in {0.5, 1.0}              (damped -> full extrapolation)
* motion_gate_on_prediction in {False, True}
  (False: gate on raw distance, motion only re-ranks the champion feasible set;
   True: gate on predicted distance, admits fast motion-consistent long links)

Writes experiments/sot2862b/screen_motion_link.json. No Kaggle submission.
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
    aggregate,
    cv_result_to_dict,
    score_family,
)
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

SIGMAS = [8.0, 15.0, 30.0]
GAINS = [0.5, 1.0]
GATES = [False, True]


def variant_link(base: LinkParams, sigma: float, gain: float, gate: bool) -> LinkParams:
    """Champion link params but with ARGUS motion-model linking enabled."""
    return LinkParams(
        max_distance=base.max_distance,
        allow_division=base.allow_division,
        division_distance=base.division_distance,
        division_max_sibling_ratio=base.division_max_sibling_ratio,
        velocity_gain=base.velocity_gain,
        velocity_disp_weight=base.velocity_disp_weight,
        motion_model_link=True,
        motion_smooth_sigma=sigma,
        motion_gain=gain,
        motion_gate_on_prediction=gate,
        max_frame_gap=base.max_frame_gap,
        gap_distance=base.gap_distance,
        min_track_length=base.min_track_length,
    )


def cv_for_link(cache: dict, link: LinkParams):
    rows = []
    for fam in CV_HOLDOUT:
        dets, gt, scale, n_true = cache[fam.name]
        pred = link_centroids(dets, scale=scale, params=link)
        rows.append(score_family(fam, pred, gt, n_true, scale=scale))
    return aggregate(rows)


def edge_counts_by_dataset(res) -> dict:
    return {
        r.name: {"tp": r.edge_tp, "fp": r.edge_fp, "fn": r.edge_fn,
                 "pred_nodes": r.num_pred_nodes}
        for r in res.per_dataset
    }


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
    base_ec = edge_counts_by_dataset(baseline)
    print(f"[baseline] champion micro_adj={baseline.micro_adj_edge_jaccard:.4f} "
          f"score={baseline.score:.4f}", flush=True)

    variants = []
    for sigma in SIGMAS:
        for gain in GAINS:
            for gate in GATES:
                link = variant_link(base_link, sigma, gain, gate)
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
                tot_d_tp = sum(v["d_tp"] for v in edge_delta.values())
                tot_d_fp = sum(v["d_fp"] for v in edge_delta.values())
                tot_d_fn = sum(v["d_fn"] for v in edge_delta.values())
                variants.append({
                    "motion_smooth_sigma": sigma,
                    "motion_gain": gain,
                    "motion_gate_on_prediction": gate,
                    "micro_adj_edge_jaccard": round(res.micro_adj_edge_jaccard, 4),
                    "micro_edge_jaccard": round(res.micro_edge_jaccard, 4),
                    "score": round(res.score, 4),
                    "delta_score_vs_champion": round(res.score - baseline.score, 4),
                    "no_per_dataset_regression": bool(no_reg),
                    "promotable": bool(no_reg and score_up),
                    "total_edge_delta": {"d_tp": tot_d_tp, "d_fp": tot_d_fp, "d_fn": tot_d_fn},
                    "per_dataset_adj": {k: round(v, 4) for k, v in per.items()},
                    "edge_delta_by_dataset": edge_delta,
                    "cv": cv_result_to_dict(res),
                })
                print(f"[variant] sigma={sigma} gain={gain} gate={gate} "
                      f"micro_adj={res.micro_adj_edge_jaccard:.4f} "
                      f"score={res.score:.4f} d={variants[-1]['delta_score_vs_champion']:+.4f} "
                      f"dTP={tot_d_tp:+d} dFP={tot_d_fp:+d} dFN={tot_d_fn:+d} "
                      f"no_reg={no_reg} promotable={variants[-1]['promotable']}", flush=True)

    variants.sort(key=lambda v: v["score"], reverse=True)
    promotable = [v for v in variants if v["promotable"]]

    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2864",
        "axis": "ARGUS motion-model predicted-position LAP linking (motion_model_link: "
                "predict every source's next position from a global spatially-smoothed "
                "motion field estimated within the frame pair, then LAP on predicted "
                "positions), single-variable ablation vs the frozen "
                "detect-link-dog-v4-shorttrack champion",
        "cv_source": "biohub_tracking.eval.cv (SOT-2761 leak-free 4-family holdout)",
        "portability_note": "cv2 (Farneback dense flow) is NOT installed in this repo/"
                            "offline kernel; the numpy/scipy-only detection-cloud motion "
                            "field is the first-class portable path (no weights/internet).",
        "baseline_note": "learned-detection baseline (SOT-2848) is degenerate (micro-adj "
                         "0.0, no family-robust operating point), so the linking axis is "
                         "measured on the champion classical detection where an A/B is "
                         "meaningful (as SOT-2849 established).",
        "distinct_from_velocity_gain": "velocity_gain (SOT-2369) predicts each cell from "
                                       "its own t-1->t edge (no prediction for history-less "
                                       "cells); the motion field predicts every source from "
                                       "its neighbourhood within the current frame pair.",
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "baseline_champion": {
            "micro_adj_edge_jaccard": round(baseline.micro_adj_edge_jaccard, 4),
            "score": round(baseline.score, 4),
            "edge_counts_by_dataset": base_ec,
            "cv": cv_result_to_dict(baseline),
        },
        "champion_per_dataset_adj_floor": CHAMPION_PER_DATASET_ADJ,
        "grid": {"motion_smooth_sigma": SIGMAS, "motion_gain": GAINS,
                 "motion_gate_on_prediction": GATES},
        "variants_ranked_by_score": variants,
        "n_promotable": len(promotable),
        "best_promotable": promotable[0] if promotable else None,
    }
    out = REPO / "experiments/sot2862b/screen_motion_link.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"baseline champion score={baseline.score:.4f} "
          f"(micro_adj={baseline.micro_adj_edge_jaccard:.4f})")
    print(f"n_promotable={len(promotable)}")
    if promotable:
        b = promotable[0]
        print(f"BEST PROMOTABLE: sigma={b['motion_smooth_sigma']} gain={b['motion_gain']} "
              f"gate={b['motion_gate_on_prediction']} score={b['score']} "
              f"(+{b['delta_score_vs_champion']}) edge_delta={b['total_edge_delta']}")
    else:
        top = variants[0]
        print(f"NO PROMOTABLE variant. Top-by-score: sigma={top['motion_smooth_sigma']} "
              f"gain={top['motion_gain']} gate={top['motion_gate_on_prediction']} "
              f"score={top['score']} (delta {top['delta_score_vs_champion']:+}) "
              f"no_reg={top['no_per_dataset_regression']} edge_delta={top['total_edge_delta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
