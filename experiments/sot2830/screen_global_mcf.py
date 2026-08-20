"""Screen the portable global min-cost-flow / birth-death-arc linker (SOT-2830).

The champion ``detect-link-dog-v4-shorttrack`` links each ``t -> t+1`` transition
with an *optimal but greedy* nearest-neighbour bipartite matching: every feasible
pair within ``max_distance`` is matched (implicit birth/death cost = infinity). The
top competition solutions instead win with **tracking-by-assignment** (ILP /
min-cost-flow; arxiv 2004.06375, 1705.03386) that jointly optimises assignment with
explicit **birth/death** arcs, so the solver can *refuse* a marginal link rather
than attach a transient detection.

This screens the portable analog (``link.global_window >= 2`` + finite
``birth_cost``/``death_cost``, scipy-only, pyscipopt-free) against the frozen
champion (single-variable same-seed A/B; the pipeline is deterministic). Because the
window min-cost flow *decouples per transition* for a pure-distance cost, the
effective lever is the link-acceptance threshold ``theta = birth_cost + death_cost``
(a pair links only when its scaled distance is ``< theta``). We sweep:

* ``theta`` in {inf (== champion sanity), 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0} µm
  (birth_cost = death_cost = theta / 2)
* ``global_window`` in {2, 3} at two thetas, to empirically confirm the flow is
  window-invariant (no cross-hop coupling) and never emits a bridge edge.

The links are **metric-valid**: only consecutive ``t -> t+1`` edges are produced, so
this does NOT reintroduce the gap-closing (SOT-2763) non-continuous-metric failure.

Detection is frozen (champion DoG-v3 adaptive ``mad_k=3.0``) and computed **once**
per family; every link variant is re-linked off the cached detections and scored
through the SOT-2817 re-anchored full-metric CV, so numbers are byte-comparable to
the registry champion CV (0.6649). Writes
``experiments/sot2830/screen_global_mcf.json``. No Kaggle submission.
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

# theta = birth_cost + death_cost (link-acceptance threshold, microns). inf == the
# champion (every feasible link kept). Below max_distance=7 it prunes the longest
# feasible links first.
THETAS = [float("inf"), 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0]
# (theta, window) pairs to confirm window-invariance / no bridge leakage.
WINDOW_INVARIANCE_CHECKS = [(5.0, 2), (5.0, 3), (4.0, 2), (4.0, 3)]


def variant_link(base: LinkParams, theta: float, window: int) -> LinkParams:
    """Champion link params but on the global birth/death-arc path."""
    half = float("inf") if theta == float("inf") else theta / 2.0
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
        global_window=window,
        birth_cost=half,
        death_cost=half,
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


def summarise(res, baseline, base_ec, theta, window):
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
        "theta": None if theta == float("inf") else theta,
        "global_window": window,
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

    # theta sweep at window=2
    variants = []
    for theta in THETAS:
        link = variant_link(base_link, theta, window=2)
        res = cv_for_link(cache, link)
        row = summarise(res, baseline, base_ec, theta, window=2)
        variants.append(row)
        tlabel = "inf" if theta == float("inf") else f"{theta:.1f}"
        print(f"[theta] theta={tlabel} w=2 micro_adj={res.micro_adj_edge_jaccard:.4f} "
              f"score={res.score:.4f} d={row['delta_score_vs_champion']:+.4f} "
              f"dTP={row['total_edge_delta']['d_tp']:+d} "
              f"dFP={row['total_edge_delta']['d_fp']:+d} "
              f"dFN={row['total_edge_delta']['d_fn']:+d} "
              f"no_reg={row['no_per_dataset_regression']} "
              f"promotable={row['promotable']}", flush=True)

    # theta=inf sanity: the global path must reproduce the champion CV exactly.
    inf_row = next(v for v in variants if v["theta"] is None)
    sanity_reproduces_champion = (
        abs(inf_row["score"] - round(baseline.score, 4)) < 1e-9
        and inf_row["total_edge_delta"] == {"d_tp": 0, "d_fp": 0, "d_fn": 0}
    )

    # window-invariance checks
    window_checks = []
    for theta, window in WINDOW_INVARIANCE_CHECKS:
        link = variant_link(base_link, theta, window=window)
        res = cv_for_link(cache, link)
        row = summarise(res, baseline, base_ec, theta, window)
        window_checks.append(row)
        print(f"[winchk] theta={theta:.1f} w={window} "
              f"micro_adj={res.micro_adj_edge_jaccard:.4f} score={res.score:.4f} "
              f"edge_delta={row['total_edge_delta']}", flush=True)

    def _same(a, b):
        return (a["score"] == b["score"]
                and a["total_edge_delta"] == b["total_edge_delta"])

    window_invariant = all(
        _same(window_checks[i], window_checks[i + 1])
        for i in (0, 2)
    )

    variants_ranked = sorted(variants, key=lambda v: v["score"], reverse=True)
    promotable = [v for v in variants_ranked if v["promotable"]]

    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2830",
        "axis": "portable global short-window min-cost-flow / birth-death-arc "
                "assignment linking (link.global_window>=2 + finite birth_cost/"
                "death_cost, scipy-only, pyscipopt-free), single-variable same-seed "
                "A/B vs the frozen detect-link-dog-v4-shorttrack champion; only "
                "consecutive t->t+1 metric-valid edges (no bridge/gap edges)",
        "cv_source": "biohub_tracking.eval.cv (SOT-2817 re-anchored full-metric "
                     "4-family leak-free holdout)",
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "lever_note": "theta = birth_cost + death_cost is the link-acceptance "
                      "threshold; a t->t+1 pair links only when scaled distance "
                      "< theta (and <= max_distance=7). theta=inf reproduces the "
                      "champion (every feasible pair matched). The window min-cost "
                      "flow decouples per transition for a pure-distance cost, so "
                      "global_window is a structural knob (window-invariant output).",
        "baseline_champion": {
            "micro_adj_edge_jaccard": round(baseline.micro_adj_edge_jaccard, 4),
            "score": round(baseline.score, 4),
            "edge_counts_by_dataset": base_ec,
            "cv": cv_result_to_dict(baseline),
        },
        "champion_per_dataset_adj_floor": CHAMPION_PER_DATASET_ADJ,
        "grid": {"theta": [None if t == float("inf") else t for t in THETAS],
                 "window": [2]},
        "theta_inf_sanity_reproduces_champion": bool(sanity_reproduces_champion),
        "window_invariance_checks": window_checks,
        "window_invariant": bool(window_invariant),
        "variants_ranked_by_score": variants_ranked,
        "n_promotable": len(promotable),
        "best_promotable": promotable[0] if promotable else None,
    }
    out = REPO / "experiments/sot2830/screen_global_mcf.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"baseline champion score={baseline.score:.4f} "
          f"(micro_adj={baseline.micro_adj_edge_jaccard:.4f})")
    print(f"theta=inf reproduces champion exactly: {sanity_reproduces_champion}")
    print(f"window_invariant: {window_invariant}")
    print(f"n_promotable={len(promotable)}")
    if promotable:
        b = promotable[0]
        print(f"BEST PROMOTABLE: theta={b['theta']} w={b['global_window']} "
              f"score={b['score']} (+{b['delta_score_vs_champion']}) "
              f"edge_delta={b['total_edge_delta']}")
    else:
        top = variants_ranked[0]
        print(f"NO PROMOTABLE variant. Top-by-score: theta={top['theta']} "
              f"score={top['score']} (delta {top['delta_score_vs_champion']:+}) "
              f"no_reg={top['no_per_dataset_regression']} "
              f"edge_delta={top['total_edge_delta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
