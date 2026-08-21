"""SOT-2910 A/B: bidirectional mutual-NN cycle-consistency edge gate vs champion.

Ports the cycle-consistency principle (NeighborTrack arXiv:2211.06663 / DistNet2D
arXiv:2310.19641): run the existing linker and keep only ``t -> t+1`` links that are
a **mutual nearest neighbour** in both directions, pruning the contested,
globally-assigned-but-not-mutual edges most likely to be an FP steal in dense
volumes. This PRUNES primary edges (adds none), so it trades recall for linker
precision — the metric-highest-leverage move (``TP/(TP+FP+FN)``, FP edges hurt most,
and a shed FP edge frees the matched GT node after the min_track_length prune).

Because the detection stage is byte-identical to the champion (only the linker knob
differs), this caches each family's detections ONCE and re-links per sweep point, so
the whole margin sweep costs one detection pass per family. Runs the SOT-2903
re-anchored leak-free CV (4 GT families, same seed / same metric): primary =
``micro_adj`` (the official royerlab metric), guardrail = ``micro_raw`` (raw
no-regression, SOT-2903). Emits a per-dataset non-regression verdict (adj AND raw)
and a promotion decision (promoted / rejected / inconclusive) plus champion
byte-freeze proof. No Kaggle submission; mutates no champion state.

Writes ``experiments/sot2910/ab_cycle_consistency.json``.
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
# Ambiguity margin sweep (scaled microns). 0.0 = pure mutual-NN prune; positive =
# also drop mutual-best links whose runner-up competitor is within the margin.
MARGIN_SWEEP = [0.0, 0.5, 1.0, 1.5, 2.0]
OUT = REPO / "experiments/sot2910/ab_cycle_consistency.json"


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

    # Detect once per family (the gate is a pure linker post-filter).
    detections_by_fam, gt_by_fam, scale_by_fam, ntrue_by_fam = {}, {}, {}, {}
    for fam in CV_HOLDOUT:
        geff = REPO / fam.geff
        arr = _open_image_array(REPO / fam.image)
        detections_by_fam[fam.name] = detect_volume_series(arr, detect_params)
        gt_by_fam[fam.name] = load_geff(geff)
        scale_by_fam[fam.name] = geff_scale(geff)
        ntrue_by_fam[fam.name] = geff_estimated_num_nodes(geff)

    # Champion reference (gate off).
    champ_cv = _cv_from_cached(
        detections_by_fam, gt_by_fam, scale_by_fam, ntrue_by_fam, champ_link
    )
    incumbent_adj = {r.name: r.adj_edge_jaccard for r in champ_cv.per_dataset}
    incumbent_raw = {r.name: r.edge_jaccard for r in champ_cv.per_dataset}
    champ_dict = cv_result_to_dict(champ_cv)

    # Single-variable sweep over the ambiguity margin (gate on).
    sweep = []
    for margin in MARGIN_SWEEP:
        link = dataclasses.replace(
            champ_link, cycle_consistency_gate=True, cycle_consistency_margin=margin
        )
        cv = _cv_from_cached(
            detections_by_fam, gt_by_fam, scale_by_fam, ntrue_by_fam, link
        )
        no_reg_adj = cv.no_regression_vs(incumbent_adj)
        no_reg_raw = _raw_no_regression(cv, incumbent_raw)
        cvd = cv_result_to_dict(cv)
        sweep.append(
            {
                "cycle_consistency_margin": margin,
                "cv": cvd,
                "no_regression_adj_vs_champion": bool(no_reg_adj),
                "no_regression_raw_vs_champion": bool(no_reg_raw),
                "delta_micro_adj": round(
                    cv.micro_adj_edge_jaccard - champ_cv.micro_adj_edge_jaccard, 4
                ),
                "delta_micro_raw": round(
                    cv.micro_edge_jaccard - champ_cv.micro_edge_jaccard, 4
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
    # strict micro-adj gain AND per-dataset adj non-regression AND that the gain is
    # not family-mix noise (macro AND lineage-macro both up) AND the raw guardrail
    # does not regress any family (SOT-2903 primary=micro_adj, guardrail=micro_raw).
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
        "issue": "SOT-2910",
        "recordedAt": datetime.now(timezone.utc).isoformat(),
        "axis": "bidirectional mutual-NN cycle-consistency edge gate (FP-edge prune)",
        "sources": [
            "https://arxiv.org/pdf/2211.06663",
            "https://arxiv.org/pdf/2310.19641",
        ],
        "champion_config_sha256": champion_sha,
        "champion_byte_frozen": champion_sha == CHAMPION_CONFIG_SHA256,
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "champion_cv": champ_dict,
        "champion_representativeness": representativeness_report(champ_cv),
        "sweep": sweep,
        "best": {
            "cycle_consistency_margin": best["cycle_consistency_margin"],
            "micro_adj": best_cv["micro_adj_edge_jaccard"],
            "micro_raw": best_cv["micro_edge_jaccard"],
            "delta_micro_adj": best_delta,
            "delta_micro_raw": best["delta_micro_raw"],
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
        f"VERDICT={verdict} best_margin={best['cycle_consistency_margin']} "
        f"delta_micro_adj={best_delta} delta_micro_raw={best['delta_micro_raw']} "
        f"no_reg_adj={best['no_regression_adj_vs_champion']} "
        f"no_reg_raw={best['no_regression_raw_vs_champion']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
