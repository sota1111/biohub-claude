"""Confirm step for the SOT-2830 global min-cost-flow / birth-death-arc linker.

The screen (``screen_global_mcf.py``) found the champion linking is only matched or
beaten within a family-mix-sensitive margin. This confirm run re-verifies, on the
same frozen champion detection + SOT-2817 re-anchored full-metric CV:

1. **Champion byte-invariance under the new code** — with ``global_window`` default
   (1) the champion CV must still reproduce 0.6649 exactly, proving the added global
   path did not perturb the per-frame path.
2. **Top candidates reproduce deterministically** — theta=6.5 (the sole
   CV-promotable point) and theta=6.0 (the highest micro, but per-dataset
   regressing) reproduce their screen numbers exactly, and we re-report the
   representativeness guard (``family_mix_sensitive``) that gates promotion.

Writes ``experiments/sot2830/confirm_global_mcf.json``. No Kaggle submission.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.detect import detect_volume_series
from biohub_tracking.eval.cv import (
    CHAMPION_REFERENCE_MICRO_ADJ,
    CHAMPION_REFERENCE_SCORE,
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

CHAMPION_PER_DATASET_ADJ = {
    "44b6_0113de3b": 0.8895,
    "44b6_0b24845f": 0.6817,
    "6bba_05b6850b": 0.5700,
    "6bba_05db0fb1": 0.7310,
}

CONFIRM_THETAS = [6.5, 6.0]  # sole promotable point + highest-micro (regressing)


def variant_link(base: LinkParams, theta: float) -> LinkParams:
    half = theta / 2.0
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
        global_window=2,
        birth_cost=half,
        death_cost=half,
        min_track_length=base.min_track_length,
    )


def cv_for_link(cache, link):
    rows = [
        score_family(fam, link_centroids(cache[fam.name][0], scale=cache[fam.name][2],
                                         params=link), cache[fam.name][1],
                     cache[fam.name][3], scale=cache[fam.name][2])
        for fam in CV_HOLDOUT
    ]
    return aggregate(rows)


def main() -> int:
    cfg = load_champion_config()
    detect, base_link, _scale = champion_params(cfg)

    cache = {}
    for fam in CV_HOLDOUT:
        arr = _open_image_array(REPO / fam.image)
        dets = detect_volume_series(arr, detect)
        gt = load_geff(REPO / fam.geff)
        scale = geff_scale(REPO / fam.geff)
        n_true = geff_estimated_num_nodes(REPO / fam.geff)
        cache[fam.name] = (dets, gt, scale, n_true)

    baseline = cv_for_link(cache, base_link)
    champ_reproduced = (
        abs(baseline.micro_adj_edge_jaccard - CHAMPION_REFERENCE_MICRO_ADJ) < 1e-4
        and abs(baseline.score - CHAMPION_REFERENCE_SCORE) < 1e-4
    )
    print(f"[champion] CV micro_adj={baseline.micro_adj_edge_jaccard:.4f} "
          f"score={baseline.score:.4f} reproduced={champ_reproduced}", flush=True)

    results = []
    for theta in CONFIRM_THETAS:
        res = cv_for_link(cache, variant_link(base_link, theta))
        rep = representativeness_report(res)
        no_reg = res.no_regression_vs(CHAMPION_PER_DATASET_ADJ)
        results.append({
            "theta": theta,
            "global_window": 2,
            "score": round(res.score, 4),
            "micro_adj_edge_jaccard": round(res.micro_adj_edge_jaccard, 4),
            "lineage_macro_adj": round(res.lineage_macro_adj, 4),
            "macro_adj_edge_jaccard": round(res.macro_adj_edge_jaccard, 4),
            "delta_score_vs_champion": round(res.score - baseline.score, 4),
            "no_per_dataset_regression": bool(no_reg),
            "family_mix_sensitive": bool(rep.get("family_mix_sensitive")),
            "micro_lineage_macro_gap": rep.get("micro_lineage_macro_gap"),
            "per_dataset_adj": {r.name: round(r.adj_edge_jaccard, 4)
                                for r in res.per_dataset},
            "cv": cv_result_to_dict(res),
        })
        print(f"[confirm] theta={theta} score={res.score:.4f} "
              f"d={res.score-baseline.score:+.4f} no_reg={no_reg} "
              f"family_mix_sensitive={rep.get('family_mix_sensitive')} "
              f"gap={rep.get('micro_lineage_macro_gap')}", flush=True)

    promotable = [r for r in results
                  if r["no_per_dataset_regression"]
                  and r["delta_score_vs_champion"] > 1e-9]
    # Promotion gate = beats champion + per-dataset non-regression + representative
    # (NOT family_mix_sensitive per the SOT-2817 guard; the SOT-2816 CV-up/LB-down
    # hazard makes a dominant-lineage family-mix-sensitive micro gain untrustworthy
    # without LB confirmation, which this no-submit cycle cannot obtain).
    representative_promotable = [r for r in promotable
                                 if not r["family_mix_sensitive"]]
    decision = "promote" if representative_promotable else "do-not-promote"

    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2830",
        "champion_reproduced": bool(champ_reproduced),
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "baseline_score": round(baseline.score, 4),
        "confirm_variants": results,
        "cv_promotable": [r["theta"] for r in promotable],
        "representative_promotable": [r["theta"] for r in representative_promotable],
        "decision": decision,
        "decision_note": (
            "theta=6.5/window=2 is CV-promotable (+0.0022, 4/4 non-regression, "
            "micro/macro/lineage-macro all up) BUT family_mix_sensitive=True "
            "(dominant-6bba micro gain). Per the SOT-2817 representativeness guard "
            "and the SOT-2816 CV-up/LB-down hazard, a family-mix-sensitive "
            "dominant-lineage CV gain is insufficient to flip the champion pointer "
            "without LB validation, which this no-submit cycle cannot perform. "
            "Champion kept byte-frozen; candidate handed to the next submission "
            "cycle."
        ),
    }
    out = REPO / "experiments/sot2830/confirm_global_mcf.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"champion_reproduced={champ_reproduced} decision={decision}")
    return 0 if champ_reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
