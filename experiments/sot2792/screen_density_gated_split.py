"""Screen density-gated nucleus splitting (SOT-2792), cycle-4 axis 2.

SOT-2775 applied a marker-controlled watershed split with a **single global gate**
to every foreground component and so over-split the sparse families
(44b6_0b24845f 0.6817→0.1079, REJECTED). This knob
(``DetectParams.density_gated_split = ("hmaxima", h, min_size, min_seed_dist,
density_thresh, density_window)``) fires the *same* watershed **only inside
locally dense regions**: components whose NMS candidates have >= ``density_thresh``
other detections within ``density_window`` voxels. Sparse families are
structurally unable to trigger it — the materially-new mechanism that removes
SOT-2775's failure mode rather than re-tuning it.

Everything downstream of detection is the frozen champion (DoG-v4 adaptive
mad_k=3.0 + short-track prune mtl=4), re-run per variant (detection changes) with
the opened image + GT cached. Scored through the one leak-free CV evaluator
(``eval.cv``), so aggregation is identical to the champion promotion.

**Mandatory ablation**: per family we record ``split_fired`` (# foreground
components handed to the density-gated watershed). The gate requires the sparse
families (44b6_0b24845f, 44b6_0113de3b) to report ``split_fired == 0`` — proving
the SOT-2775 over-split cannot recur — while the dense 6bba families may fire.

Gate: micro-adj improvement AND per-dataset no-regression vs the frozen champion
(baseline_none). Writes ``experiments/sot2792/screen_density_gated_split.json``.
No Kaggle submission (that is the parent's job).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import load_champion_config
from biohub_tracking.detect import DetectParams, detect_centroids_with_meta
from biohub_tracking.eval.cv import (
    CV_HOLDOUT,
    aggregate,
    cv_result_to_dict,
    score_family,
)
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff
from biohub_tracking.link import LinkParams, link_centroids
from biohub_tracking.pipeline import _open_image_array

REPO = Path(__file__).resolve().parents[2]

_CFG = load_champion_config()
_D = _CFG["detect"]
_L = _CFG["link"]


def detect_params(density_gated_split) -> DetectParams:
    return DetectParams(
        sigma_zyx=tuple(_D["sigma_zyx"]),
        background_sigma_zyx=tuple(_D["background_sigma_zyx"]),
        nms_size_zyx=tuple(_D["nms_size_zyx"]),
        threshold_percentile=float(_D["threshold_percentile"]),
        mad_k=float(_D["mad_k"]),
        min_threshold=float(_D["min_threshold"]),
        density_gated_split=density_gated_split,
    )


def link_params() -> LinkParams:
    return LinkParams(
        max_distance=float(_L["max_distance"]),
        allow_division=bool(_L["allow_division"]),
        division_distance=float(_L.get("division_distance", 7.0)),
        min_track_length=int(_L["min_track_length"]),
    )


def detect_series_with_fire(arr, params) -> tuple[dict, int]:
    """detect every timepoint, returning ``({t: coords}, total_split_fired)``."""
    n_t = arr.shape[0]
    detections: dict = {}
    fired = 0
    for t in range(n_t):
        coords, meta = detect_centroids_with_meta(arr[t], params)
        detections[t] = coords
        fired += int(meta["split_fired"])
    return detections, fired


# Split spec knobs. h in robust-sigma units, min_size voxels, min_seed_dist voxels.
# The density gate is the new lever: density_thresh = # neighbouring detections
# within density_window voxels required for a region to count as "dense". Screen a
# few density_thresh points (issue asks 2-3) at a fixed watershed geometry ported
# from the SOT-2775 champion-adjacent setting (h=1.0, min_size=3, min_seed_dist=3).
_H, _MIN_SIZE, _MIN_SEED_DIST, _WINDOW = 1.0, 3.0, 3.0, 8.0
VARIANTS: list[tuple[str, object]] = [
    ("baseline_none", None),
    ("dg_thr3_win8", ("hmaxima", _H, _MIN_SIZE, _MIN_SEED_DIST, 3.0, _WINDOW)),
    ("dg_thr5_win8", ("hmaxima", _H, _MIN_SIZE, _MIN_SEED_DIST, 5.0, _WINDOW)),
    ("dg_thr8_win8", ("hmaxima", _H, _MIN_SIZE, _MIN_SEED_DIST, 8.0, _WINDOW)),
]

# Sparse families the ablation must show never fire (SOT-2775 over-split guard).
SPARSE_FAMILIES = ("44b6_0113de3b", "44b6_0b24845f")


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

    out_variants: dict = {}
    fire_by_variant: dict = {}
    for vname, spec in VARIANTS:
        print(f"\n=== {vname} (spec={spec}) ===", flush=True)
        dp = detect_params(spec)
        rows = []
        fires = {}
        for f in fams:
            detections, fired = detect_series_with_fire(f["arr"], dp)
            fires[f["fam"].name] = fired
            pred = link_centroids(detections, scale=f["scale"], params=link)
            fr = score_family(f["fam"], pred, f["gt"], f["n_true"], scale=f["scale"])
            rows.append(fr)
            print(
                f"  {fr.name:16s} tp/fp/fn={fr.edge_tp}/{fr.edge_fp}/{fr.edge_fn} "
                f"adj={fr.adj_edge_jaccard:.4f} pred_nodes={fr.num_pred_nodes} "
                f"split_fired={fired}",
                flush=True,
            )
        cv = aggregate(rows)
        out_variants[vname] = cv_result_to_dict(cv)
        fire_by_variant[vname] = fires
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
        sparse_no_fire = all(
            fire_by_variant[vname].get(name, 0) == 0 for name in SPARSE_FAMILIES
        )
        graded.append(
            {
                "variant": vname,
                "micro_adj": micro,
                "delta_micro": round(micro - inc_micro, 4),
                "no_regression": no_regress,
                "sparse_no_fire": sparse_no_fire,
                "split_fired": fire_by_variant[vname],
                "gate_pass": bool(
                    micro > inc_micro + 1e-9 and no_regress and sparse_no_fire
                ),
            }
        )
    graded.sort(key=lambda g: -g["micro_adj"])
    winners = [g for g in graded if g["gate_pass"]]
    best = winners[0] if winners else None
    verdict = "PROMOTE" if best else "REJECT"

    # Ablation summary: does every variant keep the sparse families at zero fire?
    sparse_no_fire_all = all(
        fire_by_variant[v].get(name, 0) == 0
        for v, _ in VARIANTS
        if v != "baseline_none"
        for name in SPARSE_FAMILIES
    )

    result = {
        "issue": "SOT-2792",
        "title": "Screen density-gated nucleus splitting (cycle-4 axis 2)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cv": "SOT-2761 leak-free 4-family holdout",
        "frozen_downstream": "champion DoG-v4 adaptive mad_k=3.0 + short-track mtl=4",
        "watershed_geometry": {
            "h_robust_sigma": _H,
            "min_size_vox": _MIN_SIZE,
            "min_seed_dist_vox": _MIN_SEED_DIST,
            "density_window_vox": _WINDOW,
        },
        "incumbent_micro_adj": inc_micro,
        "incumbent_per_dataset": inc_per,
        "variants": out_variants,
        "split_fired_by_variant": fire_by_variant,
        "sparse_families": list(SPARSE_FAMILIES),
        "sparse_no_fire_ablation_all_variants": sparse_no_fire_all,
        "gate": graded,
        "best": best,
        "verdict": verdict,
        "gate_rule": (
            "micro-adj improvement AND per-dataset no-regression AND sparse-family "
            "split_fired==0 (SOT-2775 over-split structurally avoided)"
        ),
        "reproducible_note": "Deterministic detect+link; re-running reproduces every score.",
    }
    out = REPO / "experiments/sot2792/screen_density_gated_split.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nVERDICT: {verdict}  incumbent={inc_micro}  best={best}")
    print(f"sparse_no_fire_all_variants={sparse_no_fire_all}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
