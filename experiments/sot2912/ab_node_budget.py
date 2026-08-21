"""SOT-2912 A/B: adjusted-Jaccard node-budget operating-point calibration.

The official metric is the **adjusted** edge Jaccard — the raw Jaccard scaled by
an explicit node-budget penalty ``1 − 0.1·(N_pred − N_true)/N_true`` (royerlab
metrics.md, α=0.1). Earlier operating-point work (SOT-2789) swept the **raw**
metric and was declared exhausted there; this axis instead calibrates the
operating point against the **adjusted objective** — it targets the node-budget
penalty term directly, using the geff ``estimated_number_of_nodes`` metadata as
the per-family true-node budget ``N_true``.

Two operating-point knobs move the realised node budget:

* **min_track_length** (primary) — the champion's post-link short-track prune,
  the node-budget lever on the *linking* side. Because detection is byte-identical
  to the champion, each family is detected ONCE and re-linked per ``mtl`` value,
  so the whole primary sweep costs one detection pass per family.
* **mad_k** (secondary) — the adaptive detection threshold
  ``median + mad_k·1.4826·MAD``; raising it detects fewer nodes (a detection-side
  node-budget lever). Each ``mad_k`` value needs a re-detection pass.

Every point is scored on the SOT-2903 re-anchored leak-free CV (4 GT families,
same seed / same metric): primary = ``micro_adj`` (the official metric), guardrail
= ``micro_raw``. The core deliverable is whether the **adjusted-optimal** operating
point differs from the **raw-optimal** one (``raw_adj_divergent``) and whether any
adjusted-non-regressing point beats the champion 0.6649. No Kaggle submission;
mutates no champion state (champion/config.json stays byte-frozen).

Writes ``experiments/sot2912/ab_node_budget.json``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.detect import detect_volume_series
from biohub_tracking.eval.cv import (
    CHAMPION_REFERENCE_MICRO_ADJ,
    CV_HOLDOUT,
    aggregate,
    cv_result_to_dict,
    representativeness_report,
    score_family,
)
from biohub_tracking.eval.node_budget import (
    OperatingPoint,
    gt_node_count,
    node_budget_penalty,
    penalty_free_pred_nodes,
    select_adjusted_operating_point,
)
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff
from biohub_tracking.link import link_centroids
from biohub_tracking.pipeline import _open_image_array

REPO = Path(__file__).resolve().parents[2]
CHAMPION_CONFIG = REPO / "champion/config.json"
CHAMPION_CONFIG_SHA256 = (
    "42064648e612183e761bf9d40b70d3e8a2497453a878f1a44f5b52e410e01bdd"
)
OUT = REPO / "experiments/sot2912/ab_node_budget.json"

# Primary node-budget lever: the post-link short-track prune (linking side).
MTL_SWEEP = [1, 2, 3, 4, 5, 6, 7, 8]
CHAMPION_MTL = 4
# Secondary node-budget lever: the adaptive detection threshold (detection side).
# The champion uses mad_k=3.0; bracket it to expose the detection node budget.
MADK_SWEEP = [2.5, 3.5]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_family_inputs():
    """Load GT / scale / n_true / GT-node-count for each holdout family (disk)."""
    gt_by_fam, scale_by_fam, ntrue_by_fam, gtnodes_by_fam = {}, {}, {}, {}
    for fam in CV_HOLDOUT:
        geff = REPO / fam.geff
        gt = load_geff(geff)
        gt_by_fam[fam.name] = gt
        scale_by_fam[fam.name] = geff_scale(geff)
        ntrue_by_fam[fam.name] = geff_estimated_num_nodes(geff)
        gtnodes_by_fam[fam.name] = gt_node_count(gt)
    return gt_by_fam, scale_by_fam, ntrue_by_fam, gtnodes_by_fam


def _cv_from_detections(detections_by_fam, gt_by_fam, scale_by_fam, ntrue_by_fam, link):
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
    return all(
        fr.edge_jaccard >= incumbent_raw.get(fr.name, float("-inf"))
        for fr in cv.per_dataset
    )


def _operating_point(label, cv, incumbent_adj, incumbent_raw) -> OperatingPoint:
    return OperatingPoint(
        label=label,
        micro_adj=cv.micro_adj_edge_jaccard,
        micro_raw=cv.micro_edge_jaccard,
        macro_adj=cv.macro_adj_edge_jaccard,
        lineage_macro_adj=cv.lineage_macro_adj,
        no_regression_adj=cv.no_regression_vs(incumbent_adj),
        no_regression_raw=_raw_no_regression(cv, incumbent_raw),
        total_pred_nodes=sum(r.num_pred_nodes for r in cv.per_dataset),
        total_true_nodes=sum(
            r.n_true for r in cv.per_dataset if not _isnan(r.n_true)
        ),
    )


def _isnan(x) -> bool:
    return x != x


def _point_report(label, cv, ntrue_by_fam) -> dict:
    """Per-point CV dict plus per-family node-budget penalty diagnostics."""
    d = cv_result_to_dict(cv)
    d["label"] = label
    d["node_budget"] = [
        {
            "name": r.name,
            "pred_nodes": r.num_pred_nodes,
            "n_true": (None if _isnan(r.n_true) else r.n_true),
            "penalty_factor": (
                None
                if _isnan(node_budget_penalty(r.num_pred_nodes, r.n_true))
                else round(node_budget_penalty(r.num_pred_nodes, r.n_true), 4)
            ),
            "penalty_free_pred_nodes": (
                None if _isnan(r.n_true) else round(penalty_free_pred_nodes(r.n_true), 1)
            ),
            "over_budget": (None if _isnan(r.n_true) else bool(r.num_pred_nodes > r.n_true)),
        }
        for r in cv.per_dataset
    ]
    return d


def main() -> int:
    champ_cfg = load_champion_config(CHAMPION_CONFIG)
    detect_params, champ_link, _scale = champion_params(champ_cfg)

    gt_by_fam, scale_by_fam, ntrue_by_fam, gtnodes_by_fam = _load_family_inputs()

    # ---- Detect once at the champion detection params (primary sweep base) ----
    champ_detections = {}
    for fam in CV_HOLDOUT:
        arr = _open_image_array(REPO / fam.image)
        champ_detections[fam.name] = detect_volume_series(arr, detect_params)

    # Champion reference operating point (mtl=4, mad_k=3.0).
    champ_cv = _cv_from_detections(
        champ_detections, gt_by_fam, scale_by_fam, ntrue_by_fam, champ_link
    )
    incumbent_adj = {r.name: r.adj_edge_jaccard for r in champ_cv.per_dataset}
    incumbent_raw = {r.name: r.edge_jaccard for r in champ_cv.per_dataset}
    champ_point = _operating_point(
        f"min_track_length={CHAMPION_MTL}", champ_cv, incumbent_adj, incumbent_raw
    )

    candidates: list[OperatingPoint] = []
    sweep_reports = []

    # ---- Primary sweep: min_track_length (post-link prune, cached detections) --
    for mtl in MTL_SWEEP:
        if mtl == CHAMPION_MTL:
            cv = champ_cv  # identical to the champion; reuse (no re-link needed)
        else:
            link = dataclasses.replace(champ_link, min_track_length=mtl)
            cv = _cv_from_detections(
                champ_detections, gt_by_fam, scale_by_fam, ntrue_by_fam, link
            )
        label = f"min_track_length={mtl}"
        pt = _operating_point(label, cv, incumbent_adj, incumbent_raw)
        if mtl != CHAMPION_MTL:
            candidates.append(pt)
        sweep_reports.append(
            {"knob": "min_track_length", "value": mtl, **_point_summary(pt),
             "cv": _point_report(label, cv, ntrue_by_fam)}
        )

    # ---- Secondary sweep: mad_k detection threshold (re-detect per value) ------
    for mad_k in MADK_SWEEP:
        dparams = dataclasses.replace(detect_params, mad_k=mad_k)
        dets = {}
        for fam in CV_HOLDOUT:
            arr = _open_image_array(REPO / fam.image)
            dets[fam.name] = detect_volume_series(arr, dparams)
        # Score at the champion prune (mtl=4) so this isolates the detection lever.
        link = dataclasses.replace(champ_link, min_track_length=CHAMPION_MTL)
        cv = _cv_from_detections(dets, gt_by_fam, scale_by_fam, ntrue_by_fam, link)
        label = f"mad_k={mad_k}"
        pt = _operating_point(label, cv, incumbent_adj, incumbent_raw)
        candidates.append(pt)
        sweep_reports.append(
            {"knob": "mad_k", "value": mad_k, **_point_summary(pt),
             "cv": _point_report(label, cv, ntrue_by_fam)}
        )

    selection = select_adjusted_operating_point(candidates, champ_point)

    champion_sha = _sha256(CHAMPION_CONFIG)
    payload = {
        "issue": "SOT-2912",
        "recordedAt": datetime.now(timezone.utc).isoformat(),
        "axis": (
            "adjusted-Jaccard node-budget operating-point calibration: select the "
            "detection/min_track_length operating point that maximises the ADJUSTED "
            "edge Jaccard (node-budget-penalised) rather than the raw Jaccard"
        ),
        "sources": [
            "royerlab metrics.md (adjusted edge Jaccard, alpha=0.1)",
            "SOT-2789 (raw operating-point sweep — exhausted on the RAW metric)",
        ],
        "champion_config_sha256": champion_sha,
        "champion_byte_frozen": champion_sha == CHAMPION_CONFIG_SHA256,
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "champion_cv": cv_result_to_dict(champ_cv),
        "champion_representativeness": representativeness_report(champ_cv),
        "champion_gt_node_counts": gtnodes_by_fam,
        "champion_point": _point_summary(champ_point),
        "sweep": sweep_reports,
        "selection": {
            "best_label": selection.best.label,
            "adj_optimal_label": selection.adj_optimal_label,
            "raw_optimal_label": selection.raw_optimal_label,
            "raw_adj_divergent": selection.raw_adj_divergent,
            "best_micro_adj": round(selection.best.micro_adj, 4),
            "best_micro_raw": round(selection.best.micro_raw, 4),
            "delta_micro_adj": selection.delta_micro_adj,
            "delta_micro_raw": selection.delta_micro_raw,
            "best_no_regression_adj": selection.best.no_regression_adj,
            "best_no_regression_raw": selection.best.no_regression_raw,
        },
        "verdict": selection.verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["selection"], indent=2))
    print(
        f"VERDICT={selection.verdict} best={selection.best.label} "
        f"delta_micro_adj={selection.delta_micro_adj} "
        f"delta_micro_raw={selection.delta_micro_raw} "
        f"raw_adj_divergent={selection.raw_adj_divergent} "
        f"(raw_opt={selection.raw_optimal_label} adj_opt={selection.adj_optimal_label})"
    )
    print(f"wrote {OUT}")
    return 0


def _point_summary(pt: OperatingPoint) -> dict:
    return {
        "micro_adj": round(pt.micro_adj, 4),
        "micro_raw": round(pt.micro_raw, 4),
        "macro_adj": round(pt.macro_adj, 4),
        "lineage_macro_adj": round(pt.lineage_macro_adj, 4),
        "no_regression_adj": pt.no_regression_adj,
        "no_regression_raw": pt.no_regression_raw,
        "total_pred_nodes": pt.total_pred_nodes,
        "total_true_nodes": round(pt.total_true_nodes, 1),
    }


if __name__ == "__main__":
    raise SystemExit(main())
