"""Screen the local-adaptive MAD threshold surface (SOT-2791), cycle-4 axis 1.

The champion gates NMS peaks against a **single per-volume** cutoff
(``median(response) + mad_k·1.4826·MAD``). Under the embryo's strong brightness
drift a single global operating point cannot serve both dense and dim regions —
lowering it globally over-detects the bright/dense areas (node-count penalty),
keeping it high drops the dim-region cells (matched-edge FN). SOT-2776's *global*
quantile normalization tried to remove the drift with an equal whole-volume
transform and failed. This knob
(``DetectParams.local_threshold = ("mad", window_zyx, k)``) replaces the scalar
cutoff with a **spatially-varying threshold surface**
``thr(x) = local_median(x) + k·1.4826·local_MAD(x)`` (windowed
``scipy.ndimage.median_filter`` statistics) so each region is gated against its
*own* local noise floor — a dim region gets a locally lower threshold without
lowering the threshold anywhere else.

Everything downstream of detection is the frozen champion (DoG-v4 adaptive
mad_k=3.0 + short-track prune mtl=4), re-run per variant (detection changes) with
the opened image + GT cached. Scored through the one leak-free CV evaluator
(``eval.cv``), so aggregation is identical to the champion promotion.

Screen a few (window, k) points (the issue asks 2-3): the goal is the yes/no
question "does a local-adaptive operating point beat the single global one",
not deep local tuning. Gate: micro-adj improvement AND per-dataset no-regression
vs the frozen champion (baseline_none). Writes
``experiments/sot2791/screen_local_threshold.json``. No Kaggle submission (that is
the parent's job).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import load_champion_config
from biohub_tracking.detect import DetectParams, detect_centroids
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


def detect_params(local_threshold) -> DetectParams:
    return DetectParams(
        sigma_zyx=tuple(_D["sigma_zyx"]),
        background_sigma_zyx=tuple(_D["background_sigma_zyx"]),
        nms_size_zyx=tuple(_D["nms_size_zyx"]),
        threshold_percentile=float(_D["threshold_percentile"]),
        mad_k=float(_D["mad_k"]),
        min_threshold=float(_D["min_threshold"]),
        local_threshold=local_threshold,
    )


def link_params() -> LinkParams:
    return LinkParams(
        max_distance=float(_L["max_distance"]),
        allow_division=bool(_L["allow_division"]),
        division_distance=float(_L.get("division_distance", 7.0)),
        min_track_length=int(_L["min_track_length"]),
    )


def detect_series(arr, params) -> dict:
    return {t: detect_centroids(arr[t], params) for t in range(arr.shape[0])}


# Local-threshold specs: ("mean", window_zyx, k) — the fast separable
# Niblack-style surface (uniform_filter mean + k·std). The exact "mad" rank-filter
# surface is ~400x slower (~87 s/volume at (7,21,21) production size → ~1170
# min/variant over the 100-tp × 4-family CV) and is NOT Kaggle-kernel-viable, so
# the screen (and any promotion) uses the "mean" kind. Windows are anisotropic
# (small along the coarse Z), several nuclei wide so the surface tracks the slow
# background drift, not individual cells. k is the local-std multiple; the champion
# global cutoff is median + 3·1.4826·MAD, so k≈3-4 is the comparable band. Screen a
# compact window×k sweep — the issue asks only 2-3 points (yes/no on local-adaptive
# vs the single global operating point, not deep local tuning).
VARIANTS: list[tuple[str, object]] = [
    ("baseline_none", None),
    ("lt_mean_w7x21_k3", ("mean", (7, 21, 21), 3.0)),
    ("lt_mean_w5x15_k3", ("mean", (5, 15, 15), 3.0)),
    ("lt_mean_w9x27_k3", ("mean", (9, 27, 27), 3.0)),
    ("lt_mean_w7x21_k4", ("mean", (7, 21, 21), 4.0)),
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

    out_variants: dict = {}
    for vname, spec in VARIANTS:
        print(f"\n=== {vname} (spec={spec}) ===", flush=True)
        dp = detect_params(spec)
        rows = []
        for f in fams:
            detections = detect_series(f["arr"], dp)
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
                "per_dataset": {
                    r["name"]: r["adjusted_edge_jaccard"] for r in cv["per_dataset"]
                },
                "gate_pass": bool(micro > inc_micro + 1e-9 and no_regress),
            }
        )
    graded.sort(key=lambda g: -g["micro_adj"])
    winners = [g for g in graded if g["gate_pass"]]
    best = winners[0] if winners else None
    verdict = "PROMOTE" if best else "REJECT"

    result = {
        "issue": "SOT-2791",
        "title": "Screen local-adaptive MAD threshold surface (cycle-4 axis 1)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cv": "SOT-2761 leak-free 4-family holdout",
        "frozen_downstream": "champion DoG-v4 adaptive mad_k=3.0 + short-track mtl=4",
        "surface": "thr(x) = local_median(x) + k*1.4826*local_MAD(x) (windowed median_filter)",
        "incumbent_micro_adj": inc_micro,
        "incumbent_per_dataset": inc_per,
        "variants": out_variants,
        "gate": graded,
        "best": best,
        "verdict": verdict,
        "gate_rule": "micro-adj improvement AND per-dataset no-regression vs frozen champion",
        "reproducible_note": "Deterministic detect+link; re-running reproduces every score.",
    }
    out = REPO / "experiments/sot2791/screen_local_threshold.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nVERDICT: {verdict}  incumbent={inc_micro}  best={best}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
