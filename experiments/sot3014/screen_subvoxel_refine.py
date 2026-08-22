"""Screen intensity-weighted sub-voxel centroid refinement (SOT-3014).

A portable classical detection lever distilled from the public *xiaoleilian*
biohub classical baseline notebook
(``kaggle.com/code/xiaoleilian/biohub-cell-tracking-classical-baseline``, its
``_refine`` step). The champion DoG detector reports the **integer voxel**
coordinate of each NMS local maximum, so a nucleus whose true intensity centre
falls between voxels is quantized by up to ½ voxel per axis (Z=0.8125 µm,
XY=0.203 µm). This lever replaces each kept centroid with the intensity-weighted
centre-of-mass of the local normalized volume — **count-neutral** (adds/removes
nothing), so detection recall and the node-count penalty are untouched; only the
≤7 µm node matching and the motion-model linking can change.

Mechanistically distinct from every rejected classical detection axis, all of
which change *which/how many* detections rather than *where* an accepted one
sits: SOT-2774 multiscale-DoG, SOT-2776 quantile-norm, SOT-2775 watershed,
SOT-2793 hessian-blobness, SOT-2791 local-MAD, SOT-2792 density-split, SOT-2873
recall-recovery, SOT-2884 hypothesis-select. So this is NOT a no-new-evidence
retry of any of them.

Everything downstream of detection is the frozen champion (motion-model LAP
linking + short-track prune). Scored through the one leak-free 4-family CV
evaluator (``eval.cv``) so the aggregation is identical to the champion
promotions. Gate: micro-adj improvement AND 4/4 per-dataset no-regression vs the
frozen champion (baseline_none). Writes ``screen_subvoxel_refine.json``. No
Kaggle submission.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import load_champion_config
from biohub_tracking.detect import DetectParams, detect_volume_series
from biohub_tracking.eval.cv import CV_HOLDOUT, aggregate, cv_result_to_dict, score_family
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff
from biohub_tracking.link import LinkParams, link_centroids
from biohub_tracking.pipeline import _open_image_array

REPO = Path(__file__).resolve().parents[2]

_CFG = load_champion_config()
_D = _CFG["detect"]
_L = _CFG["link"]


def detect_params(subvoxel) -> DetectParams:
    return DetectParams(
        sigma_zyx=tuple(_D["sigma_zyx"]),
        background_sigma_zyx=tuple(_D["background_sigma_zyx"]),
        nms_size_zyx=tuple(_D["nms_size_zyx"]),
        threshold_percentile=float(_D["threshold_percentile"]),
        mad_k=float(_D["mad_k"]),
        min_threshold=float(_D["min_threshold"]),
        subvoxel_refine=subvoxel,
    )


def link_params() -> LinkParams:
    return LinkParams(
        max_distance=float(_L["max_distance"]),
        allow_division=bool(_L["allow_division"]),
        division_distance=float(_L.get("division_distance", 7.0)),
        min_track_length=int(_L["min_track_length"]),
        motion_model_link=bool(_L.get("motion_model_link", False)),
        motion_smooth_sigma=float(_L.get("motion_smooth_sigma", 15.0)),
        motion_gain=float(_L.get("motion_gain", 1.0)),
        motion_gate_on_prediction=bool(_L.get("motion_gate_on_prediction", False)),
    )


# Sweep the refinement half-window (voxels). rz small because Z is ~4x coarser.
#   xiaoleilian default is (rz=2, ryx=5); we bracket it tighter/looser.
VARIANTS: list[tuple[str, object]] = [
    ("baseline_none", None),
    ("refine_z1_yx2", (1, 2, 2)),
    ("refine_z1_yx3", (1, 3, 3)),
    ("refine_z2_yx5", (2, 5, 5)),
    ("refine_z2_yx3", (2, 3, 3)),
]


def main() -> None:
    link = link_params()

    fams = []
    for fam in CV_HOLDOUT:
        geff = REPO / fam.geff
        fams.append(
            {
                "fam": fam,
                "arr": _open_image_array(REPO / fam.image),
                "gt": load_geff(geff),
                "scale": geff_scale(geff),
                "n_true": geff_estimated_num_nodes(geff),
            }
        )
        print(f"loaded {fam.name}", flush=True)

    out_variants = {}
    for vname, subvoxel in VARIANTS:
        print(f"\n=== {vname} (subvoxel_refine={subvoxel}) ===", flush=True)
        dp = detect_params(subvoxel)
        rows = []
        for f in fams:
            detections = detect_volume_series(f["arr"], dp)
            pred = link_centroids(detections, scale=f["scale"], params=link)
            fr = score_family(f["fam"], pred, f["gt"], f["n_true"], scale=f["scale"])
            rows.append(fr)
            print(
                f"  {fr.name:16s} tp/fp/fn={fr.edge_tp}/{fr.edge_fp}/{fr.edge_fn} "
                f"adj={fr.adj_edge_jaccard:.4f} pred_nodes={fr.num_pred_nodes}",
                flush=True,
            )
        cv = aggregate(rows)
        out_variants[vname] = cv_result_to_dict(cv)
        print(
            f"  -> micro-adj={cv.micro_adj_edge_jaccard:.4f} by_lineage={cv.by_lineage}",
            flush=True,
        )

    inc = out_variants["baseline_none"]
    inc_micro = inc["micro_adj_edge_jaccard"]
    inc_per = {r["name"]: r["adjusted_edge_jaccard"] for r in inc["per_dataset"]}

    graded = []
    for vname, cv in out_variants.items():
        if vname == "baseline_none":
            continue
        micro = cv["micro_adj_edge_jaccard"]
        no_regress = all(
            r["adjusted_edge_jaccard"] >= inc_per[r["name"]] - 1e-9
            for r in cv["per_dataset"]
        )
        graded.append(
            {
                "variant": vname,
                "micro_adj": micro,
                "delta_micro": round(micro - inc_micro, 4),
                "no_regression": no_regress,
                "gate_pass": bool(micro > inc_micro + 1e-9 and no_regress),
            }
        )
    graded.sort(key=lambda g: -g["micro_adj"])
    winners = [g for g in graded if g["gate_pass"]]
    best = winners[0] if winners else None
    verdict = "PROMOTE" if best else "REJECT"

    result = {
        "issue": "SOT-3014",
        "title": "Screen intensity-weighted sub-voxel centroid refinement",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cv": "SOT-2761 leak-free 4-family holdout",
        "source_notebook": "kaggle.com/code/xiaoleilian/biohub-cell-tracking-classical-baseline (_refine)",
        "frozen_downstream": "champion DoG-v4 adaptive mad_k=3.0 + motion-link gain1 + short-track mtl=4",
        "mechanism": "count-neutral centroid position refinement (intensity-weighted CoM); distinct from all rejected count-changing axes",
        "incumbent_micro_adj": inc_micro,
        "incumbent_per_dataset": inc_per,
        "variants": out_variants,
        "gate": graded,
        "best": best,
        "verdict": verdict,
        "gate_rule": "micro-adj improvement AND per-dataset no-regression vs baseline_none",
        "reproducible_note": "Deterministic detect+link; same seed; re-running reproduces every score.",
    }
    out = REPO / "experiments/sot3014/screen_subvoxel_refine.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nVERDICT: {verdict}  incumbent={inc_micro}  best={best}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
