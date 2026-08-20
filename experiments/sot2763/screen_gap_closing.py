"""Screen gap-closing 2nd-step linking (SOT-2763) on the SOT-2761 leak-free CV.

The champion ``detect-link-dog-v4-shorttrack`` links only consecutive frames, so
a cell missed in a single frame ends its track — the dominant FN-edge source.
This screens the classical Jaqaman/TrackMate gap-closing second LAP step
(``link.max_frame_gap`` > 1, tail->head bridges within ``gap_distance`` µm) against
the frozen champion baseline (single-variable ablation; the pipeline is
deterministic, so this is a same-seed A/B) over:

* ``max_frame_gap`` in {2, 3} — max skipped frames a bridge may span
* ``gap_distance`` in {7, 10, 14} µm — absolute tail->head bridge gate

Detection is frozen (champion DoG-v3 adaptive ``mad_k=3.0``) and computed **once**
per family; every link variant is re-linked off the cached detections and scored
through the one SOT-2761 CV aggregation, so the numbers are byte-comparable to the
registry champion CV (0.6649).

Because the edge metric drops non-consecutive edges, a bridge scores nothing
directly; its lever is rescuing real short fragments from the min_track_length=4
prune (gap-close runs before prune). So we report matched-edge TP/FP/FN deltas to
tell an honest FN recovery apart from mere node-count churn.

Writes ``experiments/sot2763/screen_gap_closing.json``. No Kaggle submission.
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

MAX_FRAME_GAPS = [2, 3]
GAP_DISTANCES = [7.0, 10.0, 14.0]


def variant_link(base: LinkParams, max_frame_gap: int, gap_distance: float) -> LinkParams:
    """Champion link params but with gap-closing enabled."""
    return LinkParams(
        max_distance=base.max_distance,
        allow_division=base.allow_division,
        division_distance=base.division_distance,
        division_max_sibling_ratio=base.division_max_sibling_ratio,
        velocity_gain=base.velocity_gain,
        velocity_disp_weight=base.velocity_disp_weight,
        motion_gate_on_prediction=base.motion_gate_on_prediction,
        max_frame_gap=max_frame_gap,
        gap_distance=gap_distance,
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
    for mfg in MAX_FRAME_GAPS:
        for gd in GAP_DISTANCES:
            link = variant_link(base_link, mfg, gd)
            res = cv_for_link(cache, link)
            per = {r.name: r.adj_edge_jaccard for r in res.per_dataset}
            no_reg = res.no_regression_vs(CHAMPION_PER_DATASET_ADJ)
            score_up = res.score > baseline.score + 1e-9
            ec = edge_counts_by_dataset(res)
            # honest FN-recovery accounting vs baseline (matched-edge counts)
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
                "max_frame_gap": mfg,
                "gap_distance": gd,
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
            print(f"[variant] mfg={mfg} gd={gd} "
                  f"micro_adj={res.micro_adj_edge_jaccard:.4f} "
                  f"score={res.score:.4f} d={variants[-1]['delta_score_vs_champion']:+.4f} "
                  f"dTP={tot_d_tp:+d} dFP={tot_d_fp:+d} dFN={tot_d_fn:+d} "
                  f"no_reg={no_reg} promotable={variants[-1]['promotable']}", flush=True)

    variants.sort(key=lambda v: v["score"], reverse=True)
    promotable = [v for v in variants if v["promotable"]]

    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2763",
        "axis": "gap-closing 2nd-step linking (max_frame_gap>1 + gap_distance gate), "
                "single-variable ablation vs the frozen detect-link-dog-v4-shorttrack "
                "champion; gap-close runs before short-track prune",
        "cv_source": "biohub_tracking.eval.cv (SOT-2761 leak-free 4-family holdout)",
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "metric_note": "the edge metric drops non-consecutive edges, so a bridge "
                       "edge never scores directly; the lever is rescuing real short "
                       "fragments from the min_track_length=4 prune. total_edge_delta "
                       "(matched-edge TP/FP/FN vs champion) distinguishes honest FN "
                       "recovery from node-count churn.",
        "baseline_champion": {
            "micro_adj_edge_jaccard": round(baseline.micro_adj_edge_jaccard, 4),
            "score": round(baseline.score, 4),
            "edge_counts_by_dataset": base_ec,
            "cv": cv_result_to_dict(baseline),
        },
        "champion_per_dataset_adj_floor": CHAMPION_PER_DATASET_ADJ,
        "grid": {"max_frame_gap": MAX_FRAME_GAPS, "gap_distance": GAP_DISTANCES},
        "variants_ranked_by_score": variants,
        "n_promotable": len(promotable),
        "best_promotable": promotable[0] if promotable else None,
    }
    out = REPO / "experiments/sot2763/screen_gap_closing.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"baseline champion score={baseline.score:.4f} "
          f"(micro_adj={baseline.micro_adj_edge_jaccard:.4f})")
    print(f"n_promotable={len(promotable)}")
    if promotable:
        b = promotable[0]
        print(f"BEST PROMOTABLE: mfg={b['max_frame_gap']} gd={b['gap_distance']} "
              f"score={b['score']} (+{b['delta_score_vs_champion']}) "
              f"edge_delta={b['total_edge_delta']}")
    else:
        top = variants[0]
        print(f"NO PROMOTABLE variant. Top-by-score: mfg={top['max_frame_gap']} "
              f"gd={top['gap_distance']} score={top['score']} "
              f"(delta {top['delta_score_vs_champion']:+}) "
              f"no_reg={top['no_per_dataset_regression']} "
              f"edge_delta={top['total_edge_delta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
