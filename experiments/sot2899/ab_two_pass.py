"""SOT-2899 A/B: classical-baseline two-pass tight-then-full-gate linking vs champion.

Ports xiaoleilian's genuine public "Biohub Cell Tracking - Classical Baseline"
(LB 0.720, pure numpy/scipy) two-pass linker: Pass 1 assigns within the champion
tight gate (max_distance=7 um), Pass 2 re-links the leftovers only out to
``link_full_distance``. Because the detection stage is byte-identical to the
champion (only the linker differs), this caches each family's detections ONCE and
re-links per sweep point, so the whole sweep costs one detection pass per family.

Runs the SOT-2817 re-anchored full-metric leak-free CV (4 GT families, same seed /
same metric) for the byte-frozen champion and for each ``link_full_distance`` in a
single-variable sweep, emitting a per-dataset non-regression (4-family LOFO)
verdict and a promotion decision (promoted / rejected / inconclusive) plus champion
byte-freeze proof. No Kaggle submission; mutates no champion state.

Writes ``experiments/sot2899/ab_two_pass.json``.
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
CANDIDATE_CONFIG = REPO / "champion/candidates/sot2899-two-pass-gate.json"
CHAMPION_CONFIG_SHA256 = (
    "42064648e612183e761bf9d40b70d3e8a2497453a878f1a44f5b52e410e01bdd"
)
FULL_DISTANCE_SWEEP = [8.0, 9.0, 10.0, 11.0, 12.0, 14.0]
OUT = REPO / "experiments/sot2899/ab_two_pass.json"


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


def main() -> int:
    champ_cfg = load_champion_config(CHAMPION_CONFIG)
    cand_cfg = load_champion_config(CANDIDATE_CONFIG)
    detect_params, champ_link, _scale = champion_params(champ_cfg)
    _cd, cand_link, _cs = champion_params(cand_cfg)

    # Detect once per family (champion detect knobs == candidate detect knobs).
    detections_by_fam, gt_by_fam, scale_by_fam, ntrue_by_fam = {}, {}, {}, {}
    for fam in CV_HOLDOUT:
        geff = REPO / fam.geff
        arr = _open_image_array(REPO / fam.image)
        detections_by_fam[fam.name] = detect_volume_series(arr, detect_params)
        gt_by_fam[fam.name] = load_geff(geff)
        scale_by_fam[fam.name] = geff_scale(geff)
        ntrue_by_fam[fam.name] = geff_estimated_num_nodes(geff)

    # Champion reference (two-pass off).
    champ_cv = _cv_from_cached(
        detections_by_fam, gt_by_fam, scale_by_fam, ntrue_by_fam, champ_link
    )
    incumbent = {r.name: r.adj_edge_jaccard for r in champ_cv.per_dataset}

    # Single-variable sweep over link_full_distance (two-pass on).
    sweep = []
    for full in FULL_DISTANCE_SWEEP:
        link = dataclasses.replace(
            cand_link, link_two_pass=True, link_full_distance=full
        )
        cv = _cv_from_cached(
            detections_by_fam, gt_by_fam, scale_by_fam, ntrue_by_fam, link
        )
        no_reg = cv.no_regression_vs(incumbent)
        sweep.append(
            {
                "link_full_distance": full,
                "cv": cv_result_to_dict(cv),
                "no_regression_vs_champion": bool(no_reg),
                "delta_micro_adj": round(
                    cv.micro_adj_edge_jaccard - champ_cv.micro_adj_edge_jaccard, 4
                ),
            }
        )

    # Best non-regressing sweep point by micro-adj; else best micro-adj overall.
    non_reg = [s for s in sweep if s["no_regression_vs_champion"]]
    pool = non_reg or sweep
    best = max(pool, key=lambda s: s["cv"]["micro_adj_edge_jaccard"])
    best_delta = best["delta_micro_adj"]

    # Promotion decision (same discipline as the prior linking A/Bs): a promote
    # requires a strict micro gain AND per-dataset non-regression AND that the delta
    # is not merely family-mix noise (macro AND lineage-macro both up too).
    champ_dict = cv_result_to_dict(champ_cv)
    best_cv = best["cv"]
    macro_up = (
        best_cv["macro_adj_edge_jaccard"] is not None
        and best_cv["macro_adj_edge_jaccard"] > champ_dict["macro_adj_edge_jaccard"]
    )
    lineage_up = (
        best_cv["lineage_macro_adj"] is not None
        and best_cv["lineage_macro_adj"] > champ_dict["lineage_macro_adj"]
    )
    if best_delta > 1e-4 and best["no_regression_vs_champion"] and macro_up and lineage_up:
        verdict = "promoted"
    elif best_delta <= 1e-4 or not best["no_regression_vs_champion"]:
        verdict = "rejected"
    else:
        verdict = "inconclusive"

    champion_sha = _sha256(CHAMPION_CONFIG)
    payload = {
        "issue": "SOT-2899",
        "recordedAt": datetime.now(timezone.utc).isoformat(),
        "axis": "two-pass tight-then-full-gate Hungarian linking (classical baseline port)",
        "source": "https://www.kaggle.com/code/xiaoleilian/biohub-cell-tracking-classical-baseline",
        "champion_config_sha256": champion_sha,
        "champion_byte_frozen": champion_sha == CHAMPION_CONFIG_SHA256,
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "champion_cv": champ_dict,
        "champion_representativeness": representativeness_report(champ_cv),
        "sweep": sweep,
        "best": {
            "link_full_distance": best["link_full_distance"],
            "micro_adj": best_cv["micro_adj_edge_jaccard"],
            "delta_micro_adj": best_delta,
            "no_regression_vs_champion": best["no_regression_vs_champion"],
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
        f"VERDICT={verdict} best_full_distance={best['link_full_distance']} "
        f"delta_micro={best_delta} no_reg={best['no_regression_vs_champion']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
