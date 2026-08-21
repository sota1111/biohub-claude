"""SOT-2922 A/B: regime-conditional *linking* operating point (LOFO, leak-free).

The linking operating point — the ARGUS motion-prediction ``motion_gain``
(SOT-2900) and the bidirectional mutual-NN cycle-consistency edge prune
(``cycle_consistency_gate`` / ``cycle_consistency_margin``, SOT-2910) — is the
exact pair of levers that each help ONE density regime as a **single global**
value and were therefore rejected on the mandatory 4/4 per-dataset
non-regression gate (the family-mix / density-mix wall): mutual-NN sheds FP
steals in the dense over-detected 6bba volumes but deletes real links in the
clean sparse 44b6 family; motion_gain=2.0 is CV-superior in aggregate but held.

This axis instead makes the linking operating point **conditional on the
observable per-sequence density regime** (SOT-2921's GT-free ``median_knn_um``
covariate, computed from the champion detection point cloud). The policy — a
threshold plus a per-regime linking operating point — is fit **leave-one-family-
out (3-fit / 1-test)**: for each held-out family the threshold and the
dense/sparse operating points are chosen on the OTHER three families only
(maximising training adjusted micro Jaccard subject to per-family non-regression),
then applied to the held-out family's own observable covariate. The four
held-out predictions are aggregated into the leak-free conditional CV.

This is a **linking-only ablation**: detection is frozen at the champion params
and cached once per family; only the linking operating point varies. Independent
of the detection-side SOT-2923.

Promotion gate (same as every biohub child): conditional CV must clear **4/4
per-dataset non-regression** (adjusted AND raw) AND beat the champion micro_adj
**0.6760**. Primary = ``micro_adj`` (royerlab official adjusted edge Jaccard);
guardrail = ``micro_raw``.

Mutates no champion state (``champion/config.json`` stays byte-frozen); NO Kaggle
submission. Writes ``experiments/sot2922/ab_regime_conditional_link.json``.
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
    CV_HOLDOUT,
    aggregate,
    cv_result_to_dict,
    representativeness_report,
    score_family,
)
from biohub_tracking.eval.regime import sequence_covariates
from biohub_tracking.eval.regime_link import RegimeLinkPoint, fit_fold_policy
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff
from biohub_tracking.link import link_centroids
from biohub_tracking.pipeline import _open_image_array

REPO = Path(__file__).resolve().parents[2]
CHAMPION_CONFIG = REPO / "champion/config.json"
# Frozen champion (SOT-2909 motion-gain1) — must stay byte-identical.
CHAMPION_CONFIG_SHA256 = (
    "f2b107674d870cfd8e1b667a5d487b15b994382f9de0e9c3bc66a0c05b6522fc"
)
CHAMPION_REFERENCE_MICRO_ADJ = 0.6760
OUT = REPO / "experiments/sot2922/ab_regime_conditional_link.json"

# Linking operating-point grid. motion_gain = ARGUS motion prediction gain
# (champion 1.0; SOT-2900 reserve 2.0). cycle_consistency_gate = SOT-2910
# bidirectional mutual-NN edge prune (margin=0.0 = its best point). Champion op =
# (motion_gain=1.0, gate off). The 4-point grid is champion × {two rejected-global
# aggressive levers}, exactly the issue's dense/sparse hypothesis.
MOTION_GAIN_GRID = [1.0, 2.0]
GATE_GRID = [False, True]
CYCLE_MARGIN = 0.0
CHAMPION_OP = RegimeLinkPoint(motion_gain=1.0, cycle_consistency_gate=False, cycle_consistency_margin=0.0)
COVARIATE_KEY = "median_knn_um"  # dense_is_low: tighter spacing ⇒ denser
DENSE_IS_LOW = True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _op_grid() -> list[RegimeLinkPoint]:
    return [
        RegimeLinkPoint(motion_gain=g, cycle_consistency_gate=gate, cycle_consistency_margin=CYCLE_MARGIN)
        for g in MOTION_GAIN_GRID
        for gate in GATE_GRID
    ]


def _link_params_for(champ_link, op: RegimeLinkPoint):
    return dataclasses.replace(
        champ_link,
        motion_gain=op.motion_gain,
        cycle_consistency_gate=op.cycle_consistency_gate,
        cycle_consistency_margin=op.cycle_consistency_margin,
    )


def main() -> int:
    champ_cfg = load_champion_config(CHAMPION_CONFIG)
    detect_params, champ_link, _scale = champion_params(champ_cfg)

    # ---- Load GT / scale / n_true per family (disk) --------------------------
    gt_by_fam, scale_by_fam, ntrue_by_fam = {}, {}, {}
    for fam in CV_HOLDOUT:
        geff = REPO / fam.geff
        gt_by_fam[fam.name] = load_geff(geff)
        scale_by_fam[fam.name] = geff_scale(geff)
        ntrue_by_fam[fam.name] = geff_estimated_num_nodes(geff)

    # ---- Detect ONCE per family at champion params (linking-only ablation) ---
    dets_by_fam: dict[str, dict] = {}
    for fam in CV_HOLDOUT:
        arr = _open_image_array(REPO / fam.image)
        dets_by_fam[fam.name] = detect_volume_series(arr, detect_params)

    # ---- Observable per-sequence covariate from the CHAMPION detection cloud --
    covariate_by_fam: dict[str, float] = {}
    covariate_full: dict[str, dict] = {}
    for fam in CV_HOLDOUT:
        cov = sequence_covariates(dets_by_fam[fam.name], scale=scale_by_fam[fam.name])
        covariate_full[fam.name] = cov
        covariate_by_fam[fam.name] = float(cov[COVARIATE_KEY])

    # ---- Score the full (family, linking-operating-point) grid ---------------
    scored: dict[tuple[str, RegimeLinkPoint], object] = {}
    for op in _op_grid():
        link = _link_params_for(champ_link, op)
        for fam in CV_HOLDOUT:
            pred = link_centroids(
                dets_by_fam[fam.name], scale=scale_by_fam[fam.name], params=link
            )
            scored[(fam.name, op)] = score_family(
                fam,
                pred,
                gt_by_fam[fam.name],
                ntrue_by_fam[fam.name],
                scale=scale_by_fam[fam.name],
            )

    # ---- Champion reference CV (op = champion for every family) --------------
    champ_rows = [scored[(fam.name, CHAMPION_OP)] for fam in CV_HOLDOUT]
    champ_cv = aggregate(champ_rows)
    champ_adj_by_fam = {r.name: r.adj_edge_jaccard for r in champ_cv.per_dataset}
    champ_raw_by_fam = {r.name: r.edge_jaccard for r in champ_cv.per_dataset}

    op_grid = _op_grid()

    # ---- Leave-one-family-out conditional policy (leak-free) ----------------
    folds = []
    heldout_rows = []
    for held in CV_HOLDOUT:
        train = [f.name for f in CV_HOLDOUT if f.name != held.name]
        fit = fit_fold_policy(
            train,
            covariate_by_fam,
            scored,
            op_grid=op_grid,
            champion_op=CHAMPION_OP,
            champion_adj_by_family=champ_adj_by_fam,
            covariate_key=COVARIATE_KEY,
            dense_is_low=DENSE_IS_LOW,
        )
        held_op = fit.policy.op_for(covariate_by_fam[held.name])
        held_row = scored[(held.name, held_op)]
        heldout_rows.append(held_row)
        folds.append(
            {
                "held_out": held.name,
                "held_out_covariate": round(covariate_by_fam[held.name], 4),
                "held_out_regime": fit.policy.regime_of(covariate_by_fam[held.name]),
                "held_out_op": {
                    "motion_gain": held_op.motion_gain,
                    "cycle_consistency_gate": held_op.cycle_consistency_gate,
                    "cycle_consistency_margin": held_op.cycle_consistency_margin,
                },
                "held_out_adj": round(held_row.adj_edge_jaccard, 4),
                "champion_adj": round(champ_adj_by_fam[held.name], 4),
                "delta_adj": round(held_row.adj_edge_jaccard - champ_adj_by_fam[held.name], 4),
                "held_out_raw": round(held_row.edge_jaccard, 4),
                "champion_raw": round(champ_raw_by_fam[held.name], 4),
                "policy": fit.policy.to_dict(),
                "train_micro_adj": round(fit.train_micro_adj, 4),
                "train_no_regression": fit.train_no_regression,
                "fell_back_to_champion": fit.fell_back_to_champion,
            }
        )

    cond_cv = aggregate(heldout_rows)
    cond_adj_by_fam = {r.name: r.adj_edge_jaccard for r in cond_cv.per_dataset}
    cond_raw_by_fam = {r.name: r.edge_jaccard for r in cond_cv.per_dataset}

    no_reg_adj = all(
        cond_adj_by_fam[n] >= champ_adj_by_fam[n] - 1e-9 for n in champ_adj_by_fam
    )
    no_reg_raw = all(
        cond_raw_by_fam[n] >= champ_raw_by_fam[n] - 1e-9 for n in champ_raw_by_fam
    )
    delta_micro_adj = cond_cv.micro_adj_edge_jaccard - champ_cv.micro_adj_edge_jaccard
    delta_micro_raw = cond_cv.micro_edge_jaccard - champ_cv.micro_edge_jaccard
    beats_champ = cond_cv.micro_adj_edge_jaccard > CHAMPION_REFERENCE_MICRO_ADJ + 1e-9

    # ---- Family-mix tell: do macro_adj and lineage_macro rise together? ------
    # A genuine (non-family-mix) gain lifts micro AND macro AND lineage-macro; a
    # family-mix artifact lifts micro alone (dense lineage dominates the weight).
    family_mix_tell = {
        "champion_micro_adj": round(champ_cv.micro_adj_edge_jaccard, 4),
        "conditional_micro_adj": round(cond_cv.micro_adj_edge_jaccard, 4),
        "champion_macro_adj": round(champ_cv.macro_adj_edge_jaccard, 4),
        "conditional_macro_adj": round(cond_cv.macro_adj_edge_jaccard, 4),
        "champion_lineage_macro_adj": round(champ_cv.lineage_macro_adj, 4),
        "conditional_lineage_macro_adj": round(cond_cv.lineage_macro_adj, 4),
        "macro_and_lineage_both_up": (
            cond_cv.macro_adj_edge_jaccard > champ_cv.macro_adj_edge_jaccard + 1e-9
            and cond_cv.lineage_macro_adj
            > champ_cv.lineage_macro_adj + 1e-9
        ),
    }

    # ---- Diagnostics: per-family adjusted- vs raw-optimal op -----------------
    per_family_optima = {}
    for fam in CV_HOLDOUT:
        rows_by_op = {op: scored[(fam.name, op)] for op in op_grid}
        adj_best = max(op_grid, key=lambda o: rows_by_op[o].adj_edge_jaccard)
        raw_best = max(op_grid, key=lambda o: rows_by_op[o].edge_jaccard)
        per_family_optima[fam.name] = {
            "covariate_median_knn_um": round(covariate_by_fam[fam.name], 4),
            "pred_nodes_champion": scored[(fam.name, CHAMPION_OP)].num_pred_nodes,
            "n_true": (
                None
                if ntrue_by_fam[fam.name] != ntrue_by_fam[fam.name]
                else ntrue_by_fam[fam.name]
            ),
            "adj_optimal_op": {
                "motion_gain": adj_best.motion_gain,
                "cycle_consistency_gate": adj_best.cycle_consistency_gate,
            },
            "adj_optimal_value": round(rows_by_op[adj_best].adj_edge_jaccard, 4),
            "raw_optimal_op": {
                "motion_gain": raw_best.motion_gain,
                "cycle_consistency_gate": raw_best.cycle_consistency_gate,
            },
            "raw_optimal_value": round(rows_by_op[raw_best].edge_jaccard, 4),
            "champion_adj": round(champ_adj_by_fam[fam.name], 4),
        }

    # ---- Diagnostic: in-sample oracle (best adjusted op per family, NO LOFO) --
    # An UPPER BOUND on any covariate policy — it uses GT to pick each family's op
    # directly (leaky), so it is a diagnostic ceiling, never a promotable number.
    oracle_rows = [
        max((scored[(fam.name, op)] for op in op_grid), key=lambda r: r.adj_edge_jaccard)
        for fam in CV_HOLDOUT
    ]
    oracle_cv = aggregate(oracle_rows)

    if beats_champ and no_reg_adj and no_reg_raw:
        verdict = "promoted"
    elif delta_micro_adj > 1e-9 and not (no_reg_adj and no_reg_raw):
        verdict = "inconclusive"
    else:
        verdict = "rejected"

    champion_sha = _sha256(CHAMPION_CONFIG)
    payload = {
        "issue": "SOT-2922",
        "recordedAt": datetime.now(timezone.utc).isoformat(),
        "axis": (
            "regime-conditional linking operating point: select motion_gain "
            "(SOT-2900) + mutual-NN cycle_consistency_gate (SOT-2910) per sequence "
            "from the observable median_knn_um density covariate (SOT-2921), fit "
            "leave-one-family-out (3-fit/1-test) under the adjusted-Jaccard gate; "
            "linking-only ablation (detection frozen at champion, cached once/family)"
        ),
        "sources": [
            "SOT-2921 observable density covariate (median_knn_um, GT-free)",
            "SOT-2910 mutual-NN cycle-consistency prune (rejected global, dense-only gain)",
            "SOT-2900 motion_gain=2.0 (CV-superior, held reserve)",
            "royerlab metrics.md (adjusted edge Jaccard, alpha=0.1)",
        ],
        "champion_config_sha256": champion_sha,
        "champion_byte_frozen": champion_sha == CHAMPION_CONFIG_SHA256,
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "champion_op": {
            "motion_gain": CHAMPION_OP.motion_gain,
            "cycle_consistency_gate": CHAMPION_OP.cycle_consistency_gate,
        },
        "op_grid": {"motion_gain": MOTION_GAIN_GRID, "cycle_consistency_gate": GATE_GRID, "cycle_consistency_margin": CYCLE_MARGIN},
        "covariate_key": COVARIATE_KEY,
        "covariates": {
            k: {kk: (None if v != v else round(v, 4)) for kk, v in cov.items()}
            for k, cov in covariate_full.items()
        },
        "detection": "champion (frozen, cached once/family); linking-only ablation",
        "champion_cv": cv_result_to_dict(champ_cv),
        "champion_representativeness": representativeness_report(champ_cv),
        "conditional_cv": cv_result_to_dict(cond_cv),
        "lofo_folds": folds,
        "per_family_optima": per_family_optima,
        "family_mix_tell": family_mix_tell,
        "oracle_in_sample_cv": {
            "micro_adj": round(oracle_cv.micro_adj_edge_jaccard, 4),
            "micro_raw": round(oracle_cv.micro_edge_jaccard, 4),
            "note": "leaky upper bound (GT picks each family op); NOT promotable",
        },
        "selection": {
            "conditional_micro_adj": round(cond_cv.micro_adj_edge_jaccard, 4),
            "conditional_micro_raw": round(cond_cv.micro_edge_jaccard, 4),
            "champion_micro_adj": round(champ_cv.micro_adj_edge_jaccard, 4),
            "champion_micro_raw": round(champ_cv.micro_edge_jaccard, 4),
            "delta_micro_adj": round(delta_micro_adj, 4),
            "delta_micro_raw": round(delta_micro_raw, 4),
            "no_regression_adj": no_reg_adj,
            "no_regression_raw": no_reg_raw,
            "beats_champion_reference": beats_champ,
            "per_dataset_delta_adj": {
                n: round(cond_adj_by_fam[n] - champ_adj_by_fam[n], 4)
                for n in champ_adj_by_fam
            },
        },
        "verdict": verdict,
        "kaggle_submission": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["selection"], indent=2))
    print(json.dumps(payload["family_mix_tell"], indent=2))
    print(
        f"VERDICT={verdict} conditional_micro_adj={payload['selection']['conditional_micro_adj']} "
        f"champion={payload['selection']['champion_micro_adj']} "
        f"delta_adj={payload['selection']['delta_micro_adj']} "
        f"no_reg_adj={no_reg_adj} no_reg_raw={no_reg_raw} "
        f"oracle_upper={payload['oracle_in_sample_cv']['micro_adj']}"
    )
    print(f"champion_byte_frozen={payload['champion_byte_frozen']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
