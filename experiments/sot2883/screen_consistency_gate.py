"""Screen the Ultrack bidirectional forward↔backward motion-consistency link gate
(SOT-2883) on the SOT-2761 leak-free CV.

Axis (issue SOT-2883, cycle-8 child of SOT-2882). Ultrack (Nature Methods 2025,
royerlab) selects tracks by *temporal consistency = adjacent-frame overlap
maximization*. Ported to point detections as a **bidirectional agreement gate** layered
on the CONFIRMED-robust SOT-2864 motion-model linker (motion_model_link + gate=True,
micro-adj 0.6760, all-4-family non-regressing): a ``t -> t+1`` link ``i -> j`` is judged
by BOTH the forward field (``src[i]`` predicted -> ``dst[j]``, residual r_f) AND the SAME
field on the reversed pair (``dst[j]`` predicted -> ``src[i]``, residual r_b). Genuine
links have both small; an FP-prone link (the forward field over-smoothing onto a spurious
near neighbour) has a large residual in one direction where the backward field disagrees.
The gate PENALISES (soft weight on 0.5*(r_f+r_b)) and/or REJECTS (hard tol on either
residual) — a pure restriction of the SOT-2864 feasible set, so max_distance is preserved.

Distinct from prior linking axes (issue acceptance):
* SOT-2871 windowed running-average velocity (REJECTED, family-mix sensitive): a *carried
  single-direction* trajectory, not forward↔backward field agreement.
* SOT-2870 learned edge-gate (REJECTED): a *learned one-direction* p_edge, not a symmetric
  motion-field cross-check.

Measured on the champion classical detection (cached once per family = same-seed A/B,
byte-comparable to champion 0.6649). Baselines: (1) byte-frozen champion, (2) SOT-2864
motion reference (0.6760). Per-dataset non-regression gate vs champion is MANDATORY
(SOT-2817). Writes screen_consistency_gate.json. No Kaggle submit.
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

# SOT-2864 best pure-motion operating point (the incumbent linking result), reused as
# the field bandwidth/gain; the bidirectional gate layers on top of it.
SIGMA = 15.0
GAIN = 1.0

# Hard-gate tolerances (scaled microns) on either residual; inf == soft-only / no reject.
HARD_TOLS = [1.5, 2.5, 4.0, 6.0]
# Soft-penalty weights on the mean bidirectional residual; 0.0 == hard-only.
SOFT_WEIGHTS = [0.25, 0.5, 1.0]


def base_kwargs(base: LinkParams) -> dict:
    """Champion link knobs every motion variant inherits unchanged (SOT-2864 op point)."""
    return dict(
        max_distance=base.max_distance,
        allow_division=base.allow_division,
        division_distance=base.division_distance,
        division_max_sibling_ratio=base.division_max_sibling_ratio,
        velocity_disp_weight=base.velocity_disp_weight,
        motion_model_link=True,
        motion_gate_on_prediction=True,
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


def summarise(res, champ_baseline, motion_ref, base_ec, label, knobs):
    per = {r.name: r.adj_edge_jaccard for r in res.per_dataset}
    floor = {r.name: r.adj_edge_jaccard for r in champ_baseline.per_dataset}
    no_reg = res.no_regression_vs(floor)  # mandatory gate: vs champion
    ref_floor = {r.name: r.adj_edge_jaccard for r in motion_ref.per_dataset}
    no_reg_vs_ref = res.no_regression_vs(ref_floor)
    score_up_vs_champ = res.score > champ_baseline.score + 1e-9
    beats_ref = res.score > motion_ref.score + 1e-9
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
        "delta_score_vs_champion": round(res.score - champ_baseline.score, 4),
        "delta_score_vs_motion_ref": round(res.score - motion_ref.score, 4),
        "no_per_dataset_regression_vs_champion": bool(no_reg),
        "no_per_dataset_regression_vs_motion_ref": bool(no_reg_vs_ref),
        "family_mix_sensitive": rep["family_mix_sensitive"],
        "promotable_vs_champion": bool(no_reg and score_up_vs_champ),
        "beats_motion_ref_no_regression": bool(beats_ref and no_reg and no_reg_vs_ref),
        "total_edge_delta": tot,
        "per_dataset_adj": {k: round(v, 4) for k, v in per.items()},
        "edge_delta_by_dataset": edge_delta,
        "cv": cv_result_to_dict(res),
    }
    print(f"[{label}] micro_adj={res.micro_adj_edge_jaccard:.4f} score={res.score:.4f} "
          f"dChamp={row['delta_score_vs_champion']:+.4f} "
          f"dRef={row['delta_score_vs_motion_ref']:+.4f} "
          f"dTP={tot['d_tp']:+d} dFP={tot['d_fp']:+d} dFN={tot['d_fn']:+d} "
          f"no_reg={no_reg} mix={rep['family_mix_sensitive']} "
          f"beats_ref={row['beats_motion_ref_no_regression']}", flush=True)
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

    champ = cv_for_link(cache, base_link)
    base_ec = edge_counts(champ)
    print(f"[baseline] champion micro_adj={champ.micro_adj_edge_jaccard:.4f} "
          f"score={champ.score:.4f}", flush=True)

    # SOT-2864 motion reference (the incumbent linking result the gate must beat).
    motion_ref_link = LinkParams(**base_kwargs(base_link))
    motion_ref = cv_for_link(cache, motion_ref_link)
    ref_row = summarise(motion_ref, champ, motion_ref, base_ec,
                        "ref:motion-only(SOT-2864,gate=True)",
                        {"mechanism": "pure forward motion field, no bidirectional gate"})

    rows = [ref_row]

    # Hard-gate arm: reject links whose forward/backward residuals disagree beyond tol.
    for tol in HARD_TOLS:
        link = LinkParams(
            **base_kwargs(base_link),
            link_consistency_gate=True, link_consistency_tol=tol,
        )
        res = cv_for_link(cache, link)
        rows.append(summarise(
            res, champ, motion_ref, base_ec, f"hard tol={tol}",
            {"link_consistency_gate": True, "link_consistency_tol": tol,
             "link_consistency_weight": 0.0},
        ))

    # Soft-penalty arm: discount bidirectionally consistent successors (no hard reject).
    for w in SOFT_WEIGHTS:
        link = LinkParams(
            **base_kwargs(base_link),
            link_consistency_gate=True, link_consistency_weight=w,
        )
        res = cv_for_link(cache, link)
        rows.append(summarise(
            res, champ, motion_ref, base_ec, f"soft weight={w}",
            {"link_consistency_gate": True, "link_consistency_tol": "inf",
             "link_consistency_weight": w},
        ))

    # Combined arm: best-guess hard tol + moderate soft penalty.
    for tol in (2.5, 4.0):
        for w in (0.5,):
            link = LinkParams(
                **base_kwargs(base_link),
                link_consistency_gate=True, link_consistency_tol=tol,
                link_consistency_weight=w,
            )
            res = cv_for_link(cache, link)
            rows.append(summarise(
                res, champ, motion_ref, base_ec, f"combined tol={tol} weight={w}",
                {"link_consistency_gate": True, "link_consistency_tol": tol,
                 "link_consistency_weight": w},
            ))

    variants = [r for r in rows if r["label"] != ref_row["label"]]
    variants_sorted = sorted(variants, key=lambda v: v["score"], reverse=True)
    promotable = [v for v in variants_sorted if v["promotable_vs_champion"]]
    beats_ref = [v for v in variants_sorted if v["beats_motion_ref_no_regression"]]

    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2883",
        "axis": "Ultrack bidirectional forward<->backward motion-consistency link gate "
                "(link_consistency_gate, default-off) layered on the SOT-2864 motion "
                "linker: reuse the global smoothed motion field forward AND on the "
                "reversed frame pair; penalise/reject links whose two-direction residuals "
                "disagree. Same-seed A/B vs frozen champion (0.6649) AND motion ref (0.6760).",
        "cv_source": "biohub_tracking.eval.cv (SOT-2761 leak-free 4-family holdout)",
        "portability_note": "numpy/scipy only — reuses _motion_field_predict on the "
                            "reversed pair; no torch/attention/pretrained-weights/cv2.",
        "distinct_from_prior_axes": {
            "SOT-2871_windowed_velocity": "REJECTED, family-mix sensitive; a carried "
                "single-direction running-average velocity, NOT forward<->backward field "
                "agreement (this axis is a symmetric overlap surrogate).",
            "SOT-2870_learned_edge_gate": "REJECTED; a learned one-direction p_edge gate, "
                "NOT a symmetric motion-field cross-check.",
            "SOT-2864_motion": "+0.0111 forward-only motion field (the reference this A/B "
                "must beat with the added backward-agreement restriction).",
        },
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "baseline_champion": {
            "micro_adj_edge_jaccard": round(champ.micro_adj_edge_jaccard, 4),
            "score": round(champ.score, 4),
            "edge_counts_by_dataset": base_ec,
            "cv": cv_result_to_dict(champ),
        },
        "reference_motion_only": ref_row,
        "grid": {"hard_tols": HARD_TOLS, "soft_weights": SOFT_WEIGHTS,
                 "motion_smooth_sigma": SIGMA, "motion_gain": GAIN},
        "variants_ranked_by_score": variants_sorted,
        "n_promotable_vs_champion": len(promotable),
        "n_beats_motion_reference_no_regression": len(beats_ref),
        "best_promotable": promotable[0] if promotable else None,
        "best_beats_ref": beats_ref[0] if beats_ref else None,
    }
    out = REPO / "experiments/sot2883/screen_consistency_gate.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"baseline champion score={champ.score:.4f}")
    print(f"motion-only reference score={ref_row['score']} "
          f"(delta {ref_row['delta_score_vs_champion']:+})")
    print(f"n_promotable_vs_champion={len(promotable)} "
          f"n_beats_motion_reference={len(beats_ref)}")
    top = variants_sorted[0]
    print(f"TOP variant: {top['label']} score={top['score']} "
          f"(dChamp {top['delta_score_vs_champion']:+}, dRef {top['delta_score_vs_motion_ref']:+}) "
          f"no_reg={top['no_per_dataset_regression_vs_champion']} "
          f"mix={top['family_mix_sensitive']} beats_ref={top['beats_motion_ref_no_regression']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
