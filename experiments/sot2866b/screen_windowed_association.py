"""Screen the portable Trackastra-style windowed global association + parental-softmax
(SOT-2871) on the SOT-2761 leak-free CV.

Axis (issue SOT-2871, cycle-7 child of SOT-2866). champion links ``t -> t+1`` by a
memoryless greedy bipartite match. Trackastra (arXiv:2405.15700) instead reasons over
a short sliding window of frames. We port that portably (numpy/scipy only, no torch /
attention / pretrained weights / cv2) as a **motion-coupled LAP chain**
(``window_assoc >= 2``): each transition's assignment is solved on motion-predicted
source positions whose displacement blends the SOT-2864 global motion field with a
**carried velocity** running-averaged over the previous ``window_assoc - 1`` window
transitions. That carry is the cross-hop coupling that makes the window *bite* — the
SOT-2830 pure-distance min-cost-flow provably decoupled per transition (+0.0022,
family-mix sensitive), so the coupling is the new mechanism, not a blind retry.

Distinct from prior linking axes (issue acceptance):
* static gap-closing (SOT-2763, REJECTED — metric drops the non-consecutive bridge)
  and node-interp gap-recovery (SOT-2849, REJECTED — family-mix sensitive) touch
  missing/gap frames; this touches only the primary consecutive ``t -> t+1`` link and
  emits only metric-scored consecutive edges.
* SOT-2840/2830 global min-cost-flow decoupled per transition; the motion carry here
  couples adjacent transitions so the window is genuine joint reasoning.
* ``velocity_gain`` (SOT-2369) predicts a cell from *its own* incoming edge only; the
  carry is re-derived from the window's chosen links and blended with the global field.

parental-softmax (division constraint): one arm enables ``allow_division`` with the
windowed parental-softmax (each parent's child-association softmax-normalised so its
mass ``<= 1``; a second daughter admitted only when its share ``>= min_share``), to
measure the balanced-division-FP-suppression add-on on real data.

Measured on the **champion classical detection** (learned detection SOT-2848 is
degenerate); detection is computed once per family and cached, so every variant is a
same-seed A/B, byte-comparable to the champion CV (0.6649). Per-dataset non-regression
gate is mandatory (SOT-2817). Writes screen_windowed_association.json. No Kaggle submit.
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

# SOT-2864 best pure-motion operating point (the strongest linking result to date),
# reused as the field bandwidth/gain; the window adds the cross-hop carry on top.
SIGMA = 15.0
GAIN = 1.0

WINDOWS = [2, 3]
CARRY_WEIGHTS = [0.5, 1.0]
GATES = [False, True]


def base_kwargs(base: LinkParams) -> dict:
    """Champion link knobs that every windowed variant inherits unchanged."""
    return dict(
        max_distance=base.max_distance,
        allow_division=base.allow_division,
        division_distance=base.division_distance,
        division_max_sibling_ratio=base.division_max_sibling_ratio,
        velocity_disp_weight=base.velocity_disp_weight,
        motion_smooth_sigma=SIGMA,
        motion_gain=GAIN,
        min_track_length=base.min_track_length,
    )


def cv_for_link(cache: dict, link: LinkParams):
    rows = []
    for fam in CV_HOLDOUT:
        dets, gt, scale, n_true = cache[fam.name]
        pred = link_centroids(dets, scale=scale, params=link)
        rows.append(score_family(fam, pred, gt, n_true, scale=scale))
    return aggregate(rows)


def edge_counts(res) -> dict:
    return {
        r.name: {"tp": r.edge_tp, "fp": r.edge_fp, "fn": r.edge_fn,
                 "pred_nodes": r.num_pred_nodes}
        for r in res.per_dataset
    }


def summarise(res, baseline, base_ec, label, knobs):
    per = {r.name: r.adj_edge_jaccard for r in res.per_dataset}
    floor = {r.name: r.adj_edge_jaccard for r in baseline.per_dataset}
    no_reg = res.no_regression_vs(floor)
    score_up = res.score > baseline.score + 1e-9
    ec = edge_counts(res)
    edge_delta = {
        name: {
            "d_tp": ec[name]["tp"] - base_ec[name]["tp"],
            "d_fp": ec[name]["fp"] - base_ec[name]["fp"],
            "d_fn": ec[name]["fn"] - base_ec[name]["fn"],
            "d_pred_nodes": ec[name]["pred_nodes"] - base_ec[name]["pred_nodes"],
        }
        for name in ec
    }
    tot = {k: sum(v[k] for v in edge_delta.values()) for k in ("d_tp", "d_fp", "d_fn")}
    rep = representativeness_report(res)
    row = {
        "label": label,
        **knobs,
        "micro_adj_edge_jaccard": round(res.micro_adj_edge_jaccard, 4),
        "score": round(res.score, 4),
        "delta_score_vs_champion": round(res.score - baseline.score, 4),
        "no_per_dataset_regression": bool(no_reg),
        "family_mix_sensitive": rep["family_mix_sensitive"],
        "promotable": bool(no_reg and score_up),
        "total_edge_delta": tot,
        "per_dataset_adj": {k: round(v, 4) for k, v in per.items()},
        "edge_delta_by_dataset": edge_delta,
        "cv": cv_result_to_dict(res),
    }
    print(f"[{label}] micro_adj={res.micro_adj_edge_jaccard:.4f} score={res.score:.4f} "
          f"d={row['delta_score_vs_champion']:+.4f} dTP={tot['d_tp']:+d} "
          f"dFP={tot['d_fp']:+d} dFN={tot['d_fn']:+d} no_reg={no_reg} "
          f"mix_sensitive={rep['family_mix_sensitive']} promotable={row['promotable']}",
          flush=True)
    return row


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
    base_ec = edge_counts(baseline)
    print(f"[baseline] champion micro_adj={baseline.micro_adj_edge_jaccard:.4f} "
          f"score={baseline.score:.4f}", flush=True)

    # Reference: SOT-2864 best pure-motion (no window carry) — the incumbent linking
    # result the window must beat to justify the added coupling.
    ref = cv_for_link(cache, LinkParams(
        **base_kwargs(base_link), motion_model_link=True, motion_gate_on_prediction=True,
    ))
    ref_row = summarise(ref, baseline, base_ec, "ref:motion-only(SOT-2864,gate=True)",
                        {"mechanism": "pure motion field, no window carry"})

    rows = [ref_row]
    for W in WINDOWS:
        for cw in CARRY_WEIGHTS:
            for gate in GATES:
                link = LinkParams(
                    **base_kwargs(base_link),
                    window_assoc=W, window_carry_weight=cw,
                    motion_gate_on_prediction=gate,
                )
                res = cv_for_link(cache, link)
                rows.append(summarise(
                    res, baseline, base_ec,
                    f"W={W} carry={cw} gate={gate}",
                    {"window_assoc": W, "window_carry_weight": cw,
                     "motion_gate_on_prediction": gate, "window_theta": "inf"},
                ))

    # parental-softmax division arm (exercises the constraint on real data): W=2,
    # allow_division with the windowed parental-softmax on the best-carry setting.
    div = LinkParams(
        **{**base_kwargs(base_link), "allow_division": True},
        window_assoc=2, window_carry_weight=1.0, motion_gate_on_prediction=True,
        window_parental_softmax=True, window_softmax_min_share=0.3, window_softmax_temp=1.0,
    )
    res_div = cv_for_link(cache, div)
    rows.append(summarise(
        res_div, baseline, base_ec, "W=2 parental-softmax(allow_division)",
        {"window_assoc": 2, "window_carry_weight": 1.0, "allow_division": True,
         "window_parental_softmax": True, "window_softmax_min_share": 0.3},
    ))

    variants = [r for r in rows if r["label"] != ref_row["label"]]
    variants_sorted = sorted(variants, key=lambda v: v["score"], reverse=True)
    promotable = [v for v in variants_sorted if v["promotable"]]
    beats_ref = [v for v in variants_sorted
                 if v["score"] > ref_row["score"] + 1e-9 and v["no_per_dataset_regression"]]

    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2871",
        "axis": "portable Trackastra-style windowed global association (motion-coupled "
                "LAP chain, window_assoc>=2: carried window velocity blended with the "
                "SOT-2864 global motion field) + parental-softmax division constraint, "
                "single-mechanism A/B vs the frozen detect-link-dog-v4-shorttrack champion",
        "cv_source": "biohub_tracking.eval.cv (SOT-2761 leak-free 4-family holdout)",
        "portability_note": "numpy/scipy only — LAP chain with birth/death outlier arcs; "
                            "no torch/attention/pretrained-weights/cv2 (exec-compat green).",
        "distinct_from_prior_axes": {
            "SOT-2763_gap_closing": "REJECTED; non-consecutive bridge dropped by metric. "
                                    "This emits only consecutive t->t+1 edges.",
            "SOT-2849_gap_recover": "REJECTED; family-mix sensitive node interpolation. "
                                    "This changes the primary link cost, inserts no nodes.",
            "SOT-2840_2830_global_mcf": "+0.0022, family-mix sensitive; its pure-distance "
                                        "cost DECOUPLED per transition. The motion carry "
                                        "couples adjacent transitions so the window bites.",
            "SOT-2864_motion": "+0.0111 pure motion field, no cross-hop carry; this A/B "
                               "measures whether the windowed carry beats it.",
        },
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "baseline_champion": {
            "micro_adj_edge_jaccard": round(baseline.micro_adj_edge_jaccard, 4),
            "score": round(baseline.score, 4),
            "edge_counts_by_dataset": base_ec,
            "cv": cv_result_to_dict(baseline),
        },
        "reference_motion_only": ref_row,
        "grid": {"window_assoc": WINDOWS, "window_carry_weight": CARRY_WEIGHTS,
                 "motion_gate_on_prediction": GATES, "motion_smooth_sigma": SIGMA,
                 "motion_gain": GAIN},
        "variants_ranked_by_score": variants_sorted,
        "n_promotable_vs_champion": len(promotable),
        "n_beats_motion_reference_no_regression": len(beats_ref),
        "best_promotable": promotable[0] if promotable else None,
    }
    out = REPO / "experiments/sot2866b/screen_windowed_association.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"baseline champion score={baseline.score:.4f}")
    print(f"motion-only reference score={ref_row['score']} "
          f"(delta {ref_row['delta_score_vs_champion']:+})")
    print(f"n_promotable_vs_champion={len(promotable)} "
          f"n_beats_motion_reference={len(beats_ref)}")
    top = variants_sorted[0]
    print(f"TOP variant: {top['label']} score={top['score']} "
          f"(delta {top['delta_score_vs_champion']:+}) "
          f"no_reg={top['no_per_dataset_regression']} "
          f"mix_sensitive={top['family_mix_sensitive']} edge_delta={top['total_edge_delta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
