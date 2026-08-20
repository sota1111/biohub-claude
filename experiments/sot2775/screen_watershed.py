"""Screen watershed nucleus splitting with h-maxima seeding (SOT-2775).

The champion detector reports one centroid per NMS local maximum, so two nuclei
whose blurred DoG blobs merge into a single peak collapse to one detection — the
matched-edge FN/FP source on the *dense* ``6bba`` families. Here we swap the NMS
peak extractor for a marker-controlled watershed on the same adaptive-threshold
foreground (``DetectParams.watershed = ("hmaxima", h, min_size, min_seed_dist)``):
extended-maxima seeds (prominence >= ``h`` robust-sigma) flood their basins on the
negated response, and each basin's centroid is a detection.

We sweep the prominence gate ``h`` (robust-sigma units) and the over-split controls
(minimum basin volume ``min_size`` vox, minimum centroid spacing ``min_seed_dist``
vox) on the SOT-2761 leak-free CV holdout, watching whether the dense 6bba
matched-edge TP/FN recover **without** regressing the sparse/clean 44b6 families
(over-splitting must not fragment clean nuclei).

Everything downstream of detection is the frozen champion (DoG-v3 adaptive
``mad_k=3.0`` + short-track pruning ``min_track_length=4``). Detection is re-run per
variant because the split changes the centroids; the opened image array + GT are
cached across variants. Scored through the one leak-free CV evaluator (``eval.cv``).
Writes ``experiments/sot2775/screen_watershed.json``. No Kaggle submission.
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


def detect_params(watershed) -> DetectParams:
    return DetectParams(
        sigma_zyx=tuple(_D["sigma_zyx"]),
        background_sigma_zyx=tuple(_D["background_sigma_zyx"]),
        nms_size_zyx=tuple(_D["nms_size_zyx"]),
        threshold_percentile=float(_D["threshold_percentile"]),
        mad_k=float(_D["mad_k"]),
        min_threshold=float(_D["min_threshold"]),
        watershed=watershed,
    )


def link_params() -> LinkParams:
    return LinkParams(
        max_distance=float(_L["max_distance"]),
        allow_division=bool(_L["allow_division"]),
        division_distance=float(_L.get("division_distance", 7.0)),
        min_track_length=int(_L["min_track_length"]),
    )


# Baseline (None == champion NMS) + a focused set spanning the split/over-split
# tradeoff: a moderate prominence gate (h=2) and a conservative *surgical* variant
# (high prominence h=3 + large min basin size + a min centroid spacing) that only
# separates strongly-distinct fused nuclei — the variant most likely to recover
# dense-family TP without fragmenting the clean 44b6 families. (The aggressive
# h=1 sweep was dropped after it decisively regressed both 44b6 families —
# 0.8895->0.8691, 0.6817->0.5347 — and inflated the node count so far that linking
# became pathologically slow; that partial evidence is kept in
# ``screen_watershed_h1partial.log``.)
VARIANTS: list[tuple[str, object]] = [
    ("baseline_none", None),
    ("ws_h3_ms8_sd4", ("hmaxima", 3.0, 8.0, 4.0)),
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
    for vname, watershed in VARIANTS:
        print(f"\n=== {vname} (watershed={watershed}) ===", flush=True)
        dp = detect_params(watershed)
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
        "issue": "SOT-2775",
        "title": "Screen watershed nucleus splitting (h-maxima seeding) for dense families",
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
    out = REPO / "experiments/sot2775/screen_watershed.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nVERDICT: {verdict}  incumbent={inc_micro}  best={best}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
