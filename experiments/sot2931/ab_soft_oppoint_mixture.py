"""SOT-2931 A/B: soft per-sequence operating-point *mixture* (LOFO, leak-free).

Escalation-ladder step-4 (reformulate the problem). SOT-2922 (linking) and
SOT-2923 (detection) tried to beat the family-mix / density-mix wall with a **hard
regime label** — threshold the observable density covariate ``median_knn_um``
(SOT-2921) into dense/sparse and give each half a *discrete* operating point.
Both were REJECTED on the 4/4 per-dataset non-regression gate: a hard partition
crosscuts the family boundary (the one family that wants the aggressive prune,
``6bba_05b6850b``, is observably the sparsest, so any single threshold puts it on
the wrong side).

This axis changes the **formulation**: instead of a hard switch, form a
**continuous weighted average** of a conservative (champion: ``motion_gain=1.0``,
mutual-NN prune off) and an aggressive (``motion_gain`` up to 2.0, mutual-NN prune
on) linking operating point, weighted by ``w = σ((center − x)/scale) ∈ (0, 1)`` — a
*logistic* function of the same leak-free observable. The champion's motion gain
slides continuously toward the reserve as the observable moves; the prune switches
in only once the aggressive weight crosses a fitted activation. The SOT-2922 hard
switch is the degenerate ``scale → 0`` limit, so this strictly generalises it; any
``scale > 0`` is a non-hard-partition regime SOT-2922/2923 could not represent.

Leak-free LOFO (3-fit / 1-test): for each held-out family the mixture parameters
(``center``, ``scale``, ``gate_activation``, aggressive ``gain``) are fit on the
OTHER three families only — maximising training adjusted micro Jaccard subject to
per-family non-regression — then applied to the held-out family's own observable
covariate. The four held-out predictions aggregate into the leak-free conditional
CV. Linking-only ablation: detection frozen at champion, cached once per family;
only the linking operating point varies. The continuous blend snaps its
``motion_gain`` to a fine pre-scored grid (``GAIN_GRID``) — a disclosed numerical
discretisation of the continuum, not a two-way partition.

Promotion gate (same as every biohub child): conditional CV must clear **4/4
per-dataset non-regression** (adjusted AND raw) AND beat the champion micro_adj
**0.6760**. Primary = ``micro_adj`` (royerlab adjusted edge Jaccard); guardrail =
``micro_raw``. Mutates no champion state (``champion/config.json`` stays
byte-frozen); NO Kaggle submission. Writes
``experiments/sot2931/ab_soft_oppoint_mixture.json``.
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
from biohub_tracking.eval.regime_blend import (
    CHAMPION_ENDPOINT,
    fit_fold_policy,
    op_key,
)
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
OUT = REPO / "experiments/sot2931/ab_soft_oppoint_mixture.json"

# Pre-scored operating-point grid = GAIN_GRID × {gate off, on}. The conservative
# endpoint is the champion (1.0, gate off); the aggressive endpoint pairs a swept
# gain with the mutual-NN prune (SOT-2910, margin=0). The fine GAIN_GRID realises
# the CONTINUOUS blend: each sequence's blended motion_gain snaps to its nearest
# grid value, so the operating point moves along a continuum, not a two-way split.
GAIN_GRID = [1.0, 1.25, 1.5, 1.75, 2.0]
GATE_GRID = [False, True]
COVARIATE_KEY = "median_knn_um"  # dense_is_low: tighter spacing ⇒ denser
DENSE_IS_LOW = True

# Soft-mixture fit grids (fit leave-one-family-out on training families only).
# scale = sigmoid width in covariate microns; the near-zero control (0.01) is the
# hard-switch limit (SOT-2922) — its presence lets the fit choose hard OR soft and
# the record shows which wins. gate_activation = aggressive-weight threshold that
# turns the mutual-NN prune on (0.5 = aggressive-majority sequences; 1.0 = only
# near-fully-dense; +inf = never / pure gain blend).
SCALE_GRID = [0.01, 0.25, 0.5, 1.0, 2.0]
GATE_ACTIVATION_GRID = [0.5, 1.0, float("inf")]
AGGRESSIVE_GAIN_GRID = [1.5, 2.0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _link_params_for(champ_link, gain: float, gate: bool):
    return dataclasses.replace(
        champ_link,
        motion_gain=gain,
        cycle_consistency_gate=gate,
        cycle_consistency_margin=0.0,
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

    # ---- Score the full (family, (gain, gate)) grid --------------------------
    scored: dict[tuple[str, tuple[float, bool]], object] = {}
    for gain in GAIN_GRID:
        for gate in GATE_GRID:
            link = _link_params_for(champ_link, gain, gate)
            for fam in CV_HOLDOUT:
                pred = link_centroids(
                    dets_by_fam[fam.name], scale=scale_by_fam[fam.name], params=link
                )
                scored[(fam.name, (gain, gate))] = score_family(
                    fam,
                    pred,
                    gt_by_fam[fam.name],
                    ntrue_by_fam[fam.name],
                    scale=scale_by_fam[fam.name],
                )

    champion_op_key = op_key(CHAMPION_ENDPOINT, GAIN_GRID)  # (1.0, False)

    # ---- Champion reference CV (champion op for every family) ----------------
    champ_rows = [scored[(fam.name, champion_op_key)] for fam in CV_HOLDOUT]
    champ_cv = aggregate(champ_rows)
    champ_adj_by_fam = {r.name: r.adj_edge_jaccard for r in champ_cv.per_dataset}
    champ_raw_by_fam = {r.name: r.edge_jaccard for r in champ_cv.per_dataset}

    # ---- Leave-one-family-out soft-mixture policy (leak-free) ---------------
    folds = []
    heldout_rows = []
    for held in CV_HOLDOUT:
        train = [f.name for f in CV_HOLDOUT if f.name != held.name]
        fit = fit_fold_policy(
            train,
            covariate_by_fam,
            scored,
            gain_grid=GAIN_GRID,
            scale_grid=SCALE_GRID,
            gate_activation_grid=GATE_ACTIVATION_GRID,
            aggressive_gain_grid=AGGRESSIVE_GAIN_GRID,
            champion_adj_by_family=champ_adj_by_fam,
            covariate_key=COVARIATE_KEY,
            dense_is_low=DENSE_IS_LOW,
        )
        held_blend = fit.policy.op_for(covariate_by_fam[held.name])
        held_op = op_key(held_blend, GAIN_GRID)
        held_row = scored[(held.name, held_op)]
        heldout_rows.append(held_row)
        folds.append(
            {
                "held_out": held.name,
                "held_out_covariate": round(covariate_by_fam[held.name], 4),
                "held_out_weight": round(fit.policy.weight_of(covariate_by_fam[held.name]), 4),
                "held_out_blended_gain": round(held_blend.motion_gain, 4),
                "held_out_snapped_op": {"motion_gain": held_op[0], "cycle_consistency_gate": held_op[1]},
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
    family_mix_tell = {
        "champion_micro_adj": round(champ_cv.micro_adj_edge_jaccard, 4),
        "conditional_micro_adj": round(cond_cv.micro_adj_edge_jaccard, 4),
        "champion_macro_adj": round(champ_cv.macro_adj_edge_jaccard, 4),
        "conditional_macro_adj": round(cond_cv.macro_adj_edge_jaccard, 4),
        "champion_lineage_macro_adj": round(champ_cv.lineage_macro_adj, 4),
        "conditional_lineage_macro_adj": round(cond_cv.lineage_macro_adj, 4),
        "macro_and_lineage_both_up": (
            cond_cv.macro_adj_edge_jaccard > champ_cv.macro_adj_edge_jaccard + 1e-9
            and cond_cv.lineage_macro_adj > champ_cv.lineage_macro_adj + 1e-9
        ),
    }

    # ---- Diagnostic: in-sample soft-blend oracle (best snapped op per family) --
    # UPPER BOUND on any covariate policy — picks each family's best (gain,gate)
    # directly by GT (leaky), so it is a diagnostic ceiling, never promotable. If
    # even this ceiling cannot clear 4/4, no observable-weighted mixture can.
    oracle_rows = []
    oracle_ops = {}
    for fam in CV_HOLDOUT:
        ops = [(g, gate) for g in GAIN_GRID for gate in GATE_GRID]
        best_op = max(ops, key=lambda o: scored[(fam.name, o)].adj_edge_jaccard)
        oracle_ops[fam.name] = {"motion_gain": best_op[0], "cycle_consistency_gate": best_op[1]}
        oracle_rows.append(scored[(fam.name, best_op)])
    oracle_cv = aggregate(oracle_rows)
    oracle_no_reg = all(
        r.adj_edge_jaccard >= champ_adj_by_fam[r.name] - 1e-9 for r in oracle_rows
    )

    if beats_champ and no_reg_adj and no_reg_raw:
        verdict = "promoted"
    elif delta_micro_adj > 1e-9 and not (no_reg_adj and no_reg_raw):
        verdict = "inconclusive"
    else:
        verdict = "rejected"

    champion_sha = _sha256(CHAMPION_CONFIG)
    payload = {
        "issue": "SOT-2931",
        "recordedAt": datetime.now(timezone.utc).isoformat(),
        "axis": (
            "soft per-sequence operating-point MIXTURE: continuous w-weighted "
            "average of a conservative (champion motion_gain=1.0, prune off) and an "
            "aggressive (motion_gain up to 2.0, mutual-NN prune on) linking op, "
            "w=sigmoid((center-median_knn_um)/scale); fit leave-one-family-out "
            "(3-fit/1-test) under the adjusted-Jaccard 4/4 non-regression gate; "
            "linking-only ablation (detection frozen at champion). Soft-mixture "
            "reformulation of the REJECTED hard regime label SOT-2922/2923 "
            "(scale->0 is that hard switch; scale>0 is the new soft regime)."
        ),
        "sources": [
            "SOT-2921 observable density covariate (median_knn_um, GT-free)",
            "SOT-2922 regime-conditional linking HARD label (rejected; hard-partition crosscut)",
            "SOT-2923 regime-conditional detection HARD label (rejected)",
            "SOT-2910 mutual-NN cycle-consistency prune (rejected global, dense-only gain)",
            "SOT-2900 motion_gain=2.0 (CV-superior, held reserve)",
            "royerlab metrics.md (adjusted edge Jaccard, alpha=0.1)",
        ],
        "champion_config_sha256": champion_sha,
        "champion_byte_frozen": champion_sha == CHAMPION_CONFIG_SHA256,
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "champion_op": {"motion_gain": 1.0, "cycle_consistency_gate": False},
        "op_grid": {"motion_gain": GAIN_GRID, "cycle_consistency_gate": GATE_GRID},
        "mixture_fit_grid": {
            "scale_um": SCALE_GRID,
            "gate_activation": [g if g != float("inf") else "inf" for g in GATE_ACTIVATION_GRID],
            "aggressive_gain": AGGRESSIVE_GAIN_GRID,
            "note": "scale=0.01 is the hard-switch (SOT-2922) control; scale>0.01 is the soft mixture",
        },
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
        "family_mix_tell": family_mix_tell,
        "oracle_in_sample_cv": {
            "micro_adj": round(oracle_cv.micro_adj_edge_jaccard, 4),
            "micro_raw": round(oracle_cv.micro_edge_jaccard, 4),
            "per_family_op": oracle_ops,
            "no_regression_adj": oracle_no_reg,
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
        f"oracle_upper={payload['oracle_in_sample_cv']['micro_adj']} "
        f"oracle_no_reg={oracle_no_reg}"
    )
    print(f"champion_byte_frozen={payload['champion_byte_frozen']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
