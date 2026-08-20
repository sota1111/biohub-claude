"""Screen appearance-descriptor-augmented linking (SOT-2829).

The champion ``detect-link-dog-v4-shorttrack`` links each ``t -> t+1`` transition
on **scaled centroid distance alone**. In the dense ``6bba`` families several
plausible successors sit within ``max_distance``, so the optimal-*distance*
assignment can attach the wrong, merely nearer neighbour — costing edge TP. The
official baseline wins by matching on **appearance** (a cross-attention linker).

This screens the portable analog: a hand-crafted local intensity-patch descriptor
(:func:`biohub_tracking.detect.patch_descriptors`, pure numpy) whose cosine
similarity augments the link cost as ``dist + appearance_weight * (1 - similarity)``
(``link.appearance_weight``). The ``<= max_distance`` feasibility gate stays on the
raw distance, so appearance only re-ranks the champion's existing feasible set (no
new long-range edge; metric-valid).

Single-variable same-seed A/B vs the frozen champion detection: detection **and**
descriptors are computed **once** per family and every ``appearance_weight`` variant
re-links off the cache, scored through the SOT-2817 re-anchored full-metric CV, so
numbers are byte-comparable to the registry champion (0.6649). ``appearance_weight=0``
is the byte-invariance sanity (must reproduce the champion edge-for-edge). Writes
``experiments/sot2829/screen_appearance.json``. No Kaggle submission.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.detect import detect_volume_series_with_descriptors
from biohub_tracking.eval.cv import (
    CHAMPION_REFERENCE_MICRO_ADJ,
    CV_HOLDOUT,
    aggregate,
    cv_result_to_dict,
    representativeness_report,
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

# appearance_weight sweep (microns of penalty for a fully-dissimilar successor).
# 0.0 == champion byte-invariance sanity; the rest probe increasing appearance pull.
WEIGHTS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def variant_link(base: LinkParams, weight: float) -> LinkParams:
    """Champion link params + appearance term at ``weight``."""
    return LinkParams(
        max_distance=base.max_distance,
        allow_division=base.allow_division,
        division_distance=base.division_distance,
        division_max_sibling_ratio=base.division_max_sibling_ratio,
        velocity_gain=base.velocity_gain,
        velocity_disp_weight=base.velocity_disp_weight,
        motion_gate_on_prediction=base.motion_gate_on_prediction,
        max_frame_gap=base.max_frame_gap,
        gap_distance=base.gap_distance,
        global_window=base.global_window,
        birth_cost=base.birth_cost,
        death_cost=base.death_cost,
        appearance_weight=weight,
        min_track_length=base.min_track_length,
        division_overlay=base.division_overlay,
    )


def cv_for_link(cache: dict, link: LinkParams):
    rows = []
    for fam in CV_HOLDOUT:
        dets, descs, gt, scale, n_true = cache[fam.name]
        pred = link_centroids(dets, scale=scale, params=link, descriptors=descs)
        rows.append(score_family(fam, pred, gt, n_true, scale=scale))
    return aggregate(rows)


def edge_counts_by_dataset(res) -> dict:
    return {
        r.name: {"tp": r.edge_tp, "fp": r.edge_fp, "fn": r.edge_fn,
                 "pred_nodes": r.num_pred_nodes}
        for r in res.per_dataset
    }


def summarise(res, baseline, base_ec, weight):
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
    tot = {
        "d_tp": sum(v["d_tp"] for v in edge_delta.values()),
        "d_fp": sum(v["d_fp"] for v in edge_delta.values()),
        "d_fn": sum(v["d_fn"] for v in edge_delta.values()),
    }
    rep = representativeness_report(res)
    return {
        "appearance_weight": weight,
        "micro_adj_edge_jaccard": round(res.micro_adj_edge_jaccard, 4),
        "micro_edge_jaccard": round(res.micro_edge_jaccard, 4),
        "score": round(res.score, 4),
        "delta_score_vs_champion": round(res.score - baseline.score, 4),
        "no_per_dataset_regression": bool(no_reg),
        "promotable": bool(no_reg and score_up),
        "family_mix_sensitive": bool(rep.get("family_mix_sensitive")),
        "micro_lineage_macro_gap": rep.get("micro_lineage_macro_gap"),
        "total_edge_delta": tot,
        "per_dataset_adj": {k: round(v, 4) for k, v in per.items()},
        "edge_delta_by_dataset": edge_delta,
        "cv": cv_result_to_dict(res),
    }


def main() -> int:
    cfg = load_champion_config()
    detect, base_link, cfg_scale = champion_params(cfg)

    cache: dict = {}
    for fam in CV_HOLDOUT:
        t0 = time.time()
        arr = _open_image_array(REPO / fam.image)
        gt = load_geff(REPO / fam.geff)
        scale = geff_scale(REPO / fam.geff)
        n_true = geff_estimated_num_nodes(REPO / fam.geff)
        dets, descs = detect_volume_series_with_descriptors(arr, detect, scale=scale)
        cache[fam.name] = (dets, descs, gt, scale, n_true)
        print(f"[detect] {fam.name}: {time.time()-t0:.1f}s "
              f"dets={sum(len(v) for v in dets.values())}", flush=True)

    baseline = cv_for_link(cache, base_link)
    base_ec = edge_counts_by_dataset(baseline)
    print(f"[baseline] champion micro_adj={baseline.micro_adj_edge_jaccard:.4f} "
          f"score={baseline.score:.4f}", flush=True)

    variants = []
    for weight in WEIGHTS:
        link = variant_link(base_link, weight)
        res = cv_for_link(cache, link)
        row = summarise(res, baseline, base_ec, weight)
        variants.append(row)
        print(f"[weight] w={weight:.1f} micro_adj={res.micro_adj_edge_jaccard:.4f} "
              f"score={res.score:.4f} d={row['delta_score_vs_champion']:+.4f} "
              f"dTP={row['total_edge_delta']['d_tp']:+d} "
              f"dFP={row['total_edge_delta']['d_fp']:+d} "
              f"dFN={row['total_edge_delta']['d_fn']:+d} "
              f"no_reg={row['no_per_dataset_regression']} "
              f"promotable={row['promotable']}", flush=True)

    # weight=0 sanity: appearance term off ⇒ the champion CV reproduced exactly.
    zero_row = next(v for v in variants if v["appearance_weight"] == 0.0)
    sanity_reproduces_champion = (
        abs(zero_row["score"] - round(baseline.score, 4)) < 1e-9
        and zero_row["total_edge_delta"] == {"d_tp": 0, "d_fp": 0, "d_fn": 0}
    )

    variants_ranked = sorted(variants, key=lambda v: v["score"], reverse=True)
    promotable = [v for v in variants_ranked if v["promotable"]]

    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2829",
        "axis": "appearance-descriptor-augmented frame-to-frame linking "
                "(link.appearance_weight; dist + w*(1-cosine similarity) of a local "
                "intensity-patch descriptor, scipy/numpy-only), single-variable "
                "same-seed A/B vs the frozen detect-link-dog-v4-shorttrack champion; "
                "<=max_distance gate stays on raw distance (metric-valid, re-rank only)",
        "cv_source": "biohub_tracking.eval.cv (SOT-2817 re-anchored full-metric "
                     "4-family leak-free holdout)",
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "lever_note": "appearance_weight is the micron penalty a fully-dissimilar "
                      "successor pays; 0 reproduces the distance-only champion "
                      "byte-for-byte. Descriptor = video-standardised local patch "
                      "[mean,std,q10,q50,q90,center,contrast,grad].",
        "baseline_champion": {
            "micro_adj_edge_jaccard": round(baseline.micro_adj_edge_jaccard, 4),
            "score": round(baseline.score, 4),
            "edge_counts_by_dataset": base_ec,
            "cv": cv_result_to_dict(baseline),
        },
        "champion_per_dataset_adj_floor": CHAMPION_PER_DATASET_ADJ,
        "grid": {"appearance_weight": WEIGHTS},
        "weight_zero_sanity_reproduces_champion": bool(sanity_reproduces_champion),
        "variants_ranked_by_score": variants_ranked,
        "n_promotable": len(promotable),
        "best_promotable": promotable[0] if promotable else None,
    }
    out = REPO / "experiments/sot2829/screen_appearance.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"baseline champion score={baseline.score:.4f} "
          f"(micro_adj={baseline.micro_adj_edge_jaccard:.4f})")
    print(f"weight=0 reproduces champion exactly: {sanity_reproduces_champion}")
    print(f"n_promotable={len(promotable)}")
    if promotable:
        b = promotable[0]
        print(f"BEST PROMOTABLE: w={b['appearance_weight']} "
              f"score={b['score']} (+{b['delta_score_vs_champion']}) "
              f"edge_delta={b['total_edge_delta']}")
    else:
        top = variants_ranked[0]
        print(f"NO PROMOTABLE variant. Top-by-score: w={top['appearance_weight']} "
              f"score={top['score']} (delta {top['delta_score_vs_champion']:+}) "
              f"no_reg={top['no_per_dataset_regression']} "
              f"edge_delta={top['total_edge_delta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
