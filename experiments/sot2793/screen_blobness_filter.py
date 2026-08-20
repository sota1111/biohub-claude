"""Screen the Hessian-eigenvalue blobness precision filter (SOT-2793).

The champion detector (single-scale DoG sigma_zyx=(1,2,2)/background (2,6,6),
adaptive mad_k=3.0, NMS, short-track prune mtl=4) over-detects on the *sparse*
families (``6bba_05b6850b``: many FP candidates, node-count penalty crushes
adjusted Jaccard). This screen adds a **detection post-processing** blobness gate
(``DetectParams.blobness_filter = ("hessian", sigma_zyx, min_blobness,
max_anisotropy)``): each NMS candidate's local Hessian-of-Gaussian eigenvalues
must be all-negative (bright blob), isotropic (``a0/a2 >= min_blobness``, not a
line) and non-planar (``a2/a1 <= max_anisotropy``, not a membrane). It is the
mechanistic inverse of the REJECTED SOT-2774 scale-union max (which *added*
detections); here candidates are *pruned* to cut sparse-family FPs.

Everything downstream of detection is the frozen champion. Detection is re-run per
variant because the candidate set changes; the opened image array + GT are cached
across variants. Scored through the one leak-free CV evaluator (``eval.cv``), so
the aggregation is identical to the champion promotion. Gate: micro-adj improvement
AND per-dataset no-regression vs the frozen champion (baseline_none). Writes
``experiments/sot2793/screen_blobness_filter.json``. No Kaggle submission.
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


def detect_params(blobness) -> DetectParams:
    return DetectParams(
        sigma_zyx=tuple(_D["sigma_zyx"]),
        background_sigma_zyx=tuple(_D["background_sigma_zyx"]),
        nms_size_zyx=tuple(_D["nms_size_zyx"]),
        threshold_percentile=float(_D["threshold_percentile"]),
        mad_k=float(_D["mad_k"]),
        min_threshold=float(_D["min_threshold"]),
        blobness_filter=blobness,
    )


def link_params() -> LinkParams:
    return LinkParams(
        max_distance=float(_L["max_distance"]),
        allow_division=bool(_L["allow_division"]),
        division_distance=float(_L.get("division_distance", 7.0)),
        min_track_length=int(_L["min_track_length"]),
    )


# Hessian derivative scale = the champion detection sigma (measure the Hessian at
# the nucleus radius). Sweep the two gate thresholds:
#   min_blobness  = a0/a2 isotropy floor (higher = stricter, drops lines)
#   max_anisotropy = a2/a1 planarity cap (lower = stricter, drops membranes)
_SIG = tuple(_D["sigma_zyx"])
VARIANTS: list[tuple[str, object]] = [
    ("baseline_none", None),
    ("bf_iso005_an12", ("hessian", _SIG, 0.05, 12.0)),
    ("bf_iso010_an8", ("hessian", _SIG, 0.10, 8.0)),
    ("bf_iso015_an6", ("hessian", _SIG, 0.15, 6.0)),
    ("bf_iso020_an4", ("hessian", _SIG, 0.20, 4.0)),
    ("bf_iso010_anoff", ("hessian", _SIG, 0.10, 1e9)),
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
    for vname, blobness in VARIANTS:
        print(f"\n=== {vname} (blobness={blobness}) ===", flush=True)
        dp = detect_params(blobness)
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
        "issue": "SOT-2793",
        "title": "Screen Hessian-eigenvalue blobness precision filter",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cv": "SOT-2761 leak-free 4-family holdout",
        "frozen_downstream": "champion DoG-v4 adaptive mad_k=3.0 + short-track mtl=4",
        "hessian_sigma_zyx": list(_SIG),
        "incumbent_micro_adj": inc_micro,
        "incumbent_per_dataset": inc_per,
        "variants": out_variants,
        "gate": graded,
        "best": best,
        "verdict": verdict,
        "gate_rule": "micro-adj improvement AND per-dataset no-regression vs baseline_none",
        "reproducible_note": "Deterministic detect+link; re-running reproduces every score.",
    }
    out = REPO / "experiments/sot2793/screen_blobness_filter.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nVERDICT: {verdict}  incumbent={inc_micro}  best={best}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
