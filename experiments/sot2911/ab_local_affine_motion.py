"""SOT-2911 A/B: local neighbour-affine motion-prediction link gate vs champion.

Ports collective-motion affine analysis (PLOS Comput. Biol. pcbi.1008407 /
PMC9287486) and Amat super-voxel optical flow (Bioinformatics 29(3):373): predict
each source cell's ``t -> t+1`` position from a *per-cell local affine motion field*
fitted (ridge least squares) to its ``local_affine_k`` nearest provisional anchor
displacements, then solve the existing predicted-position LAP against those
predictions. Unlike the SOT-2864 global smoothed field (a single locally-constant
translation), the affine term captures the first-order velocity gradient (shear /
divergence) of adjacent morphogenetic patches — the mechanism that is genuinely NEW
vs the merged global motion linker.

Detection is byte-identical to the champion (only the linker knobs differ), so this
caches each family's detections ONCE and re-links per sweep point (one detection pass
per family). Runs the SOT-2903 re-anchored leak-free CV (4 GT families, same seed /
same metric): primary = ``micro_adj`` (official royerlab metric), guardrail =
``micro_raw``. Also records the SOT-2864 global motion reference (gain=2.0) on the
SAME cached detections as a differential-axis comparison. Emits a per-dataset
non-regression verdict (adj AND raw) and a promotion decision (promoted / rejected /
inconclusive) plus champion byte-freeze proof. No Kaggle submission; mutates no
champion state.

Writes ``experiments/sot2911/ab_local_affine_motion.json``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.eval.cv import (
    CHAMPION_REFERENCE_MICRO_ADJ,
    CV_HOLDOUT,
    aggregate,
    cv_result_to_dict,
    representativeness_report,
    score_family,
)
from biohub_tracking.io import (
    geff_estimated_num_nodes,
    geff_scale,
    load_geff,
)
from biohub_tracking.link import link_centroids
from biohub_tracking.pipeline import _open_image_array
from biohub_tracking.detect import detect_volume_series

REPO = Path(__file__).resolve().parents[2]
CHAMPION_CONFIG = REPO / "champion/config.json"
CHAMPION_CONFIG_SHA256 = (
    "42064648e612183e761bf9d40b70d3e8a2497453a878f1a44f5b52e410e01bdd"
)
# Local-affine knob grid. gain scales the fitted displacement (1.0 full, 2.0 mirrors
# the merged global motion reference's operating point); k = neighbour count for the
# affine fit; ridge fixed at the well-posed default.
GAIN_SWEEP = [1.0, 1.5, 2.0]
K_SWEEP = [8, 12, 20]
RIDGE = 1.0
OUT = REPO / "experiments/sot2911/ab_local_affine_motion.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cv_from_cached(detections_by_fam, gt_by_fam, scale_by_fam, ntrue_by_fam, link):
    """Score the CV by re-linking already-detected centroids with ``link``."""
    rows = []
    for fam in CV_HOLDOUT:
        pred = link_centroids(
            detections_by_fam[fam.name], scale=scale_by_fam[fam.name], params=link
        )
        rows.append(
            score_family(
                fam,
                pred,
                gt_by_fam[fam.name],
                ntrue_by_fam[fam.name],
                scale=scale_by_fam[fam.name],
            )
        )
    return aggregate(rows)


def _raw_no_regression(cv, incumbent_raw: dict[str, float]) -> bool:
    """True iff every family's RAW edge Jaccard is >= the champion's (guardrail)."""
    return all(
        fr.edge_jaccard >= incumbent_raw.get(fr.name, float("-inf"))
        for fr in cv.per_dataset
    )


def main() -> int:
    champ_cfg = load_champion_config(CHAMPION_CONFIG)
    detect_params, champ_link, _scale = champion_params(champ_cfg)

    # Detect once per family (the affine prediction is a pure linker-side change).
    detections_by_fam, gt_by_fam, scale_by_fam, ntrue_by_fam = {}, {}, {}, {}
    for fam in CV_HOLDOUT:
        geff = REPO / fam.geff
        arr = _open_image_array(REPO / fam.image)
        detections_by_fam[fam.name] = detect_volume_series(arr, detect_params)
        gt_by_fam[fam.name] = load_geff(geff)
        scale_by_fam[fam.name] = geff_scale(geff)
        ntrue_by_fam[fam.name] = geff_estimated_num_nodes(geff)

    # Champion reference (all prediction knobs off).
    champ_cv = _cv_from_cached(
        detections_by_fam, gt_by_fam, scale_by_fam, ntrue_by_fam, champ_link
    )
    incumbent_adj = {r.name: r.adj_edge_jaccard for r in champ_cv.per_dataset}
    incumbent_raw = {r.name: r.edge_jaccard for r in champ_cv.per_dataset}
    champ_dict = cv_result_to_dict(champ_cv)

    # SOT-2864 global motion reference (gain=2.0) on the SAME cached detections, for a
    # differential-axis record (does the local affine term add over the global field?).
    global_link = dataclasses.replace(
        champ_link, motion_model_link=True, motion_gain=2.0
    )
    global_cv = _cv_from_cached(
        detections_by_fam, gt_by_fam, scale_by_fam, ntrue_by_fam, global_link
    )
    global_dict = cv_result_to_dict(global_cv)

    # Grid sweep over (gain, k) for the local affine prediction (gate re-ranks only;
    # motion_gate_on_prediction stays at champion default = raw-distance feasibility).
    sweep = []
    for gain in GAIN_SWEEP:
        for k in K_SWEEP:
            link = dataclasses.replace(
                champ_link,
                local_affine_predict=True,
                local_affine_gain=gain,
                local_affine_k=k,
                local_affine_ridge=RIDGE,
            )
            cv = _cv_from_cached(
                detections_by_fam, gt_by_fam, scale_by_fam, ntrue_by_fam, link
            )
            no_reg_adj = cv.no_regression_vs(incumbent_adj)
            no_reg_raw = _raw_no_regression(cv, incumbent_raw)
            cvd = cv_result_to_dict(cv)
            sweep.append(
                {
                    "local_affine_gain": gain,
                    "local_affine_k": k,
                    "local_affine_ridge": RIDGE,
                    "cv": cvd,
                    "no_regression_adj_vs_champion": bool(no_reg_adj),
                    "no_regression_raw_vs_champion": bool(no_reg_raw),
                    "delta_micro_adj": round(
                        cv.micro_adj_edge_jaccard - champ_cv.micro_adj_edge_jaccard, 4
                    ),
                    "delta_micro_raw": round(
                        cv.micro_edge_jaccard - champ_cv.micro_edge_jaccard, 4
                    ),
                    "delta_micro_adj_vs_global": round(
                        cv.micro_adj_edge_jaccard - global_cv.micro_adj_edge_jaccard, 4
                    ),
                }
            )

    # Best non-regressing (adj) sweep point by micro-adj; else best micro-adj overall.
    non_reg = [s for s in sweep if s["no_regression_adj_vs_champion"]]
    pool = non_reg or sweep
    best = max(pool, key=lambda s: s["cv"]["micro_adj_edge_jaccard"])
    best_delta = best["delta_micro_adj"]
    best_cv = best["cv"]

    # Promotion discipline (same as the prior linking A/Bs): a promote requires a
    # strict micro-adj gain AND per-dataset adj non-regression AND that the gain is not
    # family-mix noise (macro AND lineage-macro both up) AND the raw guardrail does not
    # regress any family (SOT-2903 primary=micro_adj, guardrail=micro_raw).
    macro_up = (
        best_cv["macro_adj_edge_jaccard"] is not None
        and best_cv["macro_adj_edge_jaccard"] > champ_dict["macro_adj_edge_jaccard"]
    )
    lineage_up = (
        best_cv["lineage_macro_adj"] is not None
        and best_cv["lineage_macro_adj"] > champ_dict["lineage_macro_adj"]
    )
    if (
        best_delta > 1e-4
        and best["no_regression_adj_vs_champion"]
        and best["no_regression_raw_vs_champion"]
        and macro_up
        and lineage_up
    ):
        verdict = "promoted"
    elif best_delta <= 1e-4 or not best["no_regression_adj_vs_champion"]:
        verdict = "rejected"
    else:
        verdict = "inconclusive"

    champion_sha = _sha256(CHAMPION_CONFIG)
    payload = {
        "issue": "SOT-2911",
        "recordedAt": datetime.now(timezone.utc).isoformat(),
        "axis": "local neighbour-affine motion-prediction link gate (velocity-gradient)",
        "sources": [
            "https://doi.org/10.1371/journal.pcbi.1008407",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9287486/",
            "https://doi.org/10.1093/bioinformatics/bts706",
        ],
        "champion_config_sha256": champion_sha,
        "champion_byte_frozen": champion_sha == CHAMPION_CONFIG_SHA256,
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "champion_cv": champ_dict,
        "champion_representativeness": representativeness_report(champ_cv),
        "global_motion_reference_gain2": {
            "micro_adj": global_dict["micro_adj_edge_jaccard"],
            "micro_raw": global_dict["micro_edge_jaccard"],
            "delta_micro_adj_vs_champion": round(
                global_cv.micro_adj_edge_jaccard - champ_cv.micro_adj_edge_jaccard, 4
            ),
            "no_regression_adj_vs_champion": bool(
                global_cv.no_regression_vs(incumbent_adj)
            ),
        },
        "sweep": sweep,
        "best": {
            "local_affine_gain": best["local_affine_gain"],
            "local_affine_k": best["local_affine_k"],
            "local_affine_ridge": best["local_affine_ridge"],
            "micro_adj": best_cv["micro_adj_edge_jaccard"],
            "micro_raw": best_cv["micro_edge_jaccard"],
            "delta_micro_adj": best_delta,
            "delta_micro_raw": best["delta_micro_raw"],
            "delta_micro_adj_vs_global": best["delta_micro_adj_vs_global"],
            "no_regression_adj_vs_champion": best["no_regression_adj_vs_champion"],
            "no_regression_raw_vs_champion": best["no_regression_raw_vs_champion"],
            "macro_up": bool(macro_up),
            "lineage_macro_up": bool(lineage_up),
        },
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT}")
    print(
        f"VERDICT={verdict} best_gain={best['local_affine_gain']} "
        f"best_k={best['local_affine_k']} delta_micro_adj={best_delta} "
        f"delta_micro_raw={best['delta_micro_raw']} "
        f"delta_vs_global={best['delta_micro_adj_vs_global']} "
        f"no_reg_adj={best['no_regression_adj_vs_champion']} "
        f"no_reg_raw={best['no_regression_raw_vs_champion']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
