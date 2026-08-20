"""Screen per-volume robust quantile intensity normalization (SOT-2776).

royerlab's public baseline contrast-stretches each volume before detection. This
repo's detector feeds raw intensity to the DoG, so the adaptive MAD threshold
drifts across the embryo's strong brightness change and drops dim family/timepoint
cells (higher edge FN). Here we prepend a per-volume ``[plow, phigh]`` percentile
clip → ``[0, 1]`` rescale (``DetectParams.intensity_norm``) and sweep the
percentile band on the SOT-2761 leak-free CV holdout, watching whether dim-family
recall (edge FN) recovers **without** per-dataset regression.

Everything downstream of detection is the frozen champion (DoG-v3 adaptive
``mad_k=3.0`` + short-track pruning ``min_track_length=4``). The MAD threshold is
recomputed on the post-normalization response automatically (it is derived inside
``detect_centroids``). Detection is re-run per variant because normalization
changes the response; the opened image array + GT are cached across variants.

Scored through the one leak-free CV evaluator (``eval.cv``), so the aggregation is
identical to the champion promotion. Writes
``experiments/sot2776/screen_quantile_norm.json``. No Kaggle submission.
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

# Frozen champion detect/link, minus the intensity_norm knob we sweep.
_CFG = load_champion_config()
_D = _CFG["detect"]
_L = _CFG["link"]


def detect_params(norm) -> DetectParams:
    return DetectParams(
        sigma_zyx=tuple(_D["sigma_zyx"]),
        background_sigma_zyx=tuple(_D["background_sigma_zyx"]),
        nms_size_zyx=tuple(_D["nms_size_zyx"]),
        threshold_percentile=float(_D["threshold_percentile"]),
        mad_k=float(_D["mad_k"]),
        min_threshold=float(_D["min_threshold"]),
        intensity_norm=norm,
    )


def link_params() -> LinkParams:
    return LinkParams(
        max_distance=float(_L["max_distance"]),
        allow_division=bool(_L["allow_division"]),
        division_distance=float(_L.get("division_distance", 7.0)),
        min_track_length=int(_L["min_track_length"]),
    )


# Baseline (None == champion) + a robust-quantile grid. Symmetric bands of
# increasing aggressiveness, plus one asymmetric band that keeps the bright tail.
VARIANTS: list[tuple[str, object]] = [
    ("baseline_none", None),
    ("q_p1_p99", ("quantile", 1.0, 99.0)),
    ("q_p0.5_p99.5", ("quantile", 0.5, 99.5)),
    ("q_p2_p98", ("quantile", 2.0, 98.0)),
    ("q_p1_p99.9", ("quantile", 1.0, 99.9)),
    ("q_p5_p95", ("quantile", 5.0, 95.0)),
]


def main() -> None:
    link = link_params()

    # Cache opened arrays + GT/scale/n_true per family (invariant across variants).
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
    for vname, norm in VARIANTS:
        print(f"\n=== {vname} (norm={norm}) ===", flush=True)
        dp = detect_params(norm)
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
        "issue": "SOT-2776",
        "title": "Screen per-volume robust quantile intensity normalization before DoG",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cv": "SOT-2761 leak-free 4-family holdout",
        "frozen_downstream": "champion DoG-v3 adaptive mad_k=3.0 + short-track mtl=4",
        "incumbent_micro_adj": inc_micro,
        "incumbent_per_dataset": inc_per,
        "variants": out_variants,
        "gate": graded,
        "best": best,
        "verdict": verdict,
        "gate_rule": "micro-adj improvement AND per-dataset no-regression vs baseline_none",
        "reproducible_note": "Deterministic detect+link; re-running reproduces every score.",
    }
    out = REPO / "experiments/sot2776/screen_quantile_norm.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nVERDICT: {verdict}  incumbent={inc_micro}  best={best}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
