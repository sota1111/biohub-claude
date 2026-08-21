"""SOT-2920 A/B: per-track constant-velocity Kalman Mahalanobis link gate vs champion.

Ports the TrackMate/trackpy constant-velocity Kalman tracker to pure numpy/scipy:
each track keeps a ``[z, y, x, vz, vy, vx]`` state, predicts its next position and
*innovation covariance* ``S``, and the ``t -> t+1`` LAP is gated/costed by the
**Mahalanobis** distance ``sqrt((z-mu)ᵀ S⁻¹ (z-mu))`` — a per-track,
uncertainty-normalised gate replacing the champion's fixed 7 µm Euclidean radius
(:func:`biohub_tracking.link._kalman_link`). Evaluated as a **single axis with
``motion_model_link`` OFF** so its per-track *temporal* state is separated from the
champion's global smoothed *spatial* motion field (the double-counting the issue warns
against): the headline A/B is champion (byte-frozen, motion field ON) vs Kalman axis
(motion field OFF). A *differential* reference — the motion-OFF nearest-neighbour
baseline — is scored on the SAME cached detections so we can tell whether the per-track
Kalman state adds a separable signal over pure NN, or is merely a worse substitute for
the global field (→ non-regression FAIL → rejected, per the issue's contract).

Detection is byte-identical to the champion (linker-only change), so each family's
detections are computed ONCE and re-linked per sweep point. Runs the SOT-2903
re-anchored leak-free CV (4 GT families, same seed / same metric): primary =
``micro_adj`` (official royerlab metric), guardrail = ``micro_raw``. Emits a per-dataset
4/4 non-regression verdict (adj AND raw) and a promotion decision (promoted / rejected /
inconclusive) plus champion byte-freeze proof. No Kaggle submission; mutates no champion
state.

Writes ``experiments/sot2920/ab_kalman_gate.json``.
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
    # Current champion (SOT-2909 motion-model-link gain=1.0 promotion, CV micro_adj 0.6760).
    "f2b107674d870cfd8e1b667a5d487b15b994382f9de0e9c3bc66a0c05b6522fc"
)
# Kalman knob grid. process_noise q inflates the predicted covariance (wider, softer
# gate → trusts the CV prediction less, toward the Euclidean champion); gate_chi2 is the
# Mahalanobis² hard gate (inf = re-rank only; 11.345 = χ²_0.99 3-DOF; 7.815 = χ²_0.95).
Q_SWEEP = [0.5, 1.0, 2.0]
CHI2_SWEEP = [float("inf"), 11.345]
OBS_NOISE = 1.0
INIT_POS_VAR = 1.0
INIT_VEL_VAR = 100.0
OUT = REPO / "experiments/sot2920/ab_kalman_gate.json"


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

    # Detect once per family (the Kalman gate is a pure linker-side change).
    detections_by_fam, gt_by_fam, scale_by_fam, ntrue_by_fam = {}, {}, {}, {}
    for fam in CV_HOLDOUT:
        geff = REPO / fam.geff
        arr = _open_image_array(REPO / fam.image)
        detections_by_fam[fam.name] = detect_volume_series(arr, detect_params)
        gt_by_fam[fam.name] = load_geff(geff)
        scale_by_fam[fam.name] = geff_scale(geff)
        ntrue_by_fam[fam.name] = geff_estimated_num_nodes(geff)

    # Headline reference: the byte-frozen champion (global motion field ON).
    champ_cv = _cv_from_cached(
        detections_by_fam, gt_by_fam, scale_by_fam, ntrue_by_fam, champ_link
    )
    incumbent_adj = {r.name: r.adj_edge_jaccard for r in champ_cv.per_dataset}
    incumbent_raw = {r.name: r.edge_jaccard for r in champ_cv.per_dataset}
    champ_dict = cv_result_to_dict(champ_cv)

    # Differential reference: motion field OFF, no per-track state (pure NN). Isolates
    # whether the Kalman per-track state adds a *separable* signal over plain NN, or is
    # only substituting (worse) for the global field the champion already has.
    nn_link = dataclasses.replace(
        champ_link, motion_model_link=False, kalman_gate=False
    )
    nn_cv = _cv_from_cached(
        detections_by_fam, gt_by_fam, scale_by_fam, ntrue_by_fam, nn_link
    )
    nn_dict = cv_result_to_dict(nn_cv)

    # Grid sweep over (process_noise, gate_chi2) for the Kalman gate. SINGLE AXIS:
    # motion_model_link OFF so the per-track temporal state is evaluated alone.
    sweep = []
    for q in Q_SWEEP:
        for chi2 in CHI2_SWEEP:
            link = dataclasses.replace(
                champ_link,
                motion_model_link=False,  # single-axis isolation (issue mandate)
                kalman_gate=True,
                kalman_process_noise=q,
                kalman_obs_noise=OBS_NOISE,
                kalman_gate_chi2=chi2,
                kalman_init_pos_var=INIT_POS_VAR,
                kalman_init_vel_var=INIT_VEL_VAR,
            )
            cv = _cv_from_cached(
                detections_by_fam, gt_by_fam, scale_by_fam, ntrue_by_fam, link
            )
            no_reg_adj = cv.no_regression_vs(incumbent_adj)
            no_reg_raw = _raw_no_regression(cv, incumbent_raw)
            cvd = cv_result_to_dict(cv)
            sweep.append(
                {
                    "kalman_process_noise": q,
                    "kalman_gate_chi2": (None if chi2 == float("inf") else chi2),
                    "kalman_obs_noise": OBS_NOISE,
                    "kalman_init_vel_var": INIT_VEL_VAR,
                    "cv": cvd,
                    "no_regression_adj_vs_champion": bool(no_reg_adj),
                    "no_regression_raw_vs_champion": bool(no_reg_raw),
                    "delta_micro_adj_vs_champion": round(
                        cv.micro_adj_edge_jaccard - champ_cv.micro_adj_edge_jaccard, 4
                    ),
                    "delta_micro_raw_vs_champion": round(
                        cv.micro_edge_jaccard - champ_cv.micro_edge_jaccard, 4
                    ),
                    "delta_micro_adj_vs_nn": round(
                        cv.micro_adj_edge_jaccard - nn_cv.micro_adj_edge_jaccard, 4
                    ),
                }
            )

    # Best non-regressing (adj) sweep point by micro-adj; else best micro-adj overall.
    non_reg = [s for s in sweep if s["no_regression_adj_vs_champion"]]
    pool = non_reg or sweep
    best = max(pool, key=lambda s: s["cv"]["micro_adj_edge_jaccard"])
    best_delta = best["delta_micro_adj_vs_champion"]
    best_cv = best["cv"]

    # Promotion discipline (identical to the prior linking A/Bs): a promote requires a
    # strict micro-adj gain vs the champion AND per-dataset adj non-regression AND that
    # the gain is not family-mix noise (macro AND lineage-macro both up) AND the raw
    # guardrail does not regress any family (SOT-2903 primary=micro_adj, guardrail=raw).
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
        "issue": "SOT-2920",
        "recordedAt": datetime.now(timezone.utc).isoformat(),
        "axis": "per-track constant-velocity Kalman innovation-covariance Mahalanobis link gate",
        "single_axis_motion_model_link": False,
        "sources": [
            "TrackMate (Tinevez et al., Methods 2017) constant-velocity Kalman tracker",
            "trackpy predict module (Crocker-Grier linking with velocity prediction)",
        ],
        "champion_config_sha256": champion_sha,
        "champion_byte_frozen": champion_sha == CHAMPION_CONFIG_SHA256,
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "champion_cv": champ_dict,
        "champion_representativeness": representativeness_report(champ_cv),
        "nn_baseline_motion_off": {
            "micro_adj": nn_dict["micro_adj_edge_jaccard"],
            "micro_raw": nn_dict["micro_edge_jaccard"],
            "delta_micro_adj_vs_champion": round(
                nn_cv.micro_adj_edge_jaccard - champ_cv.micro_adj_edge_jaccard, 4
            ),
        },
        "sweep": sweep,
        "best": {
            "kalman_process_noise": best["kalman_process_noise"],
            "kalman_gate_chi2": best["kalman_gate_chi2"],
            "micro_adj": best_cv["micro_adj_edge_jaccard"],
            "micro_raw": best_cv["micro_edge_jaccard"],
            "delta_micro_adj_vs_champion": best_delta,
            "delta_micro_raw_vs_champion": best["delta_micro_raw_vs_champion"],
            "delta_micro_adj_vs_nn": best["delta_micro_adj_vs_nn"],
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
        f"VERDICT={verdict} best_q={best['kalman_process_noise']} "
        f"best_chi2={best['kalman_gate_chi2']} "
        f"delta_micro_adj_vs_champion={best_delta} "
        f"delta_micro_raw={best['delta_micro_raw_vs_champion']} "
        f"delta_vs_nn={best['delta_micro_adj_vs_nn']} "
        f"no_reg_adj={best['no_regression_adj_vs_champion']} "
        f"no_reg_raw={best['no_regression_raw_vs_champion']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
