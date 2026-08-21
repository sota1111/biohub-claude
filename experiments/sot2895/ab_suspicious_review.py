"""SOT-2895 A/B: post-hoc suspicious-edge-review gate vs champion on leak-free CV.

Runs the SOT-2894 re-anchored full-metric leak-free CV (4 GT families, same seed /
same metric) for:

* the byte-frozen champion (``champion/config.json``), and
* the candidate (``champion/candidates/sot2895-suspicious-edge-review.json``) which
  is byte-identical to the champion EXCEPT ``suspicious_review=true`` (default-off
  post-hoc gate ported from dalloliogm's public "Biohub Suspicious Tracking Event
  Review" notebook).

Emits a per-dataset non-regression (4-family LOFO) verdict and a promotion decision
(promoted / rejected / inconclusive) plus champion byte-freeze proof. No Kaggle
submission; mutates no champion state.

Writes ``experiments/sot2895/ab_suspicious_review.json``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.eval.cv import (
    CHAMPION_REFERENCE_MICRO_ADJ,
    CHAMPION_REFERENCE_SCORE,
    cv_result_to_dict,
    evaluate_cv,
    representativeness_report,
)

REPO = Path(__file__).resolve().parents[2]
CHAMPION_CONFIG = REPO / "champion/config.json"
CANDIDATE_CONFIG = REPO / "champion/candidates/sot2895-suspicious-edge-review.json"
CHAMPION_CONFIG_SHA256 = (
    "42064648e612183e761bf9d40b70d3e8a2497453a878f1a44f5b52e410e01bdd"
)
# Noise band: promote only when micro gain exceeds this (documented CV noise ~0.005).
NOISE_BAND = 0.005


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    champ_cfg = load_champion_config(CHAMPION_CONFIG)
    cand_cfg = json.loads(CANDIDATE_CONFIG.read_text())

    # Prove the candidate flips ONLY the post-hoc gate (detection + core link knobs
    # identical to the champion; this is a post-processing-only change).
    champ_detect, champ_link, _ = champion_params(champ_cfg)
    cand_detect, cand_link, _ = champion_params(cand_cfg)
    effective = {
        "suspicious_review": cand_link.suspicious_review,
        "suspicious_turn_cos": cand_link.suspicious_turn_cos,
        "suspicious_jump_ratio": cand_link.suspicious_jump_ratio,
        "suspicious_jump_floor": cand_link.suspicious_jump_floor,
        "max_distance": cand_link.max_distance,
        "min_track_length": cand_link.min_track_length,
        "allow_division": cand_link.allow_division,
        "mad_k": cand_detect.mad_k,
        "threshold_percentile": cand_detect.threshold_percentile,
    }
    postproc_only = (
        cand_link.suspicious_review is True
        and champ_link.suspicious_review is False
        and cand_detect.mad_k == champ_detect.mad_k
        and cand_detect.threshold_percentile == champ_detect.threshold_percentile
        and cand_link.max_distance == champ_link.max_distance
        and cand_link.min_track_length == champ_link.min_track_length
        and cand_link.allow_division == champ_link.allow_division
        and cand_link.motion_model_link == champ_link.motion_model_link
    )

    champ = evaluate_cv(config=champ_cfg)
    cand = evaluate_cv(config=cand_cfg)

    champ_per_ds = {r.name: round(r.adj_edge_jaccard, 4) for r in champ.per_dataset}
    cand_per_ds = {r.name: round(r.adj_edge_jaccard, 4) for r in cand.per_dataset}
    delta_per_ds = {
        k: round(cand_per_ds[k] - champ_per_ds[k], 4) for k in champ_per_ds
    }
    no_regression = cand.no_regression_vs(champ_per_ds)

    d_micro = cand.micro_adj_edge_jaccard - champ.micro_adj_edge_jaccard
    d_macro = cand.macro_adj_edge_jaccard - champ.macro_adj_edge_jaccard
    d_lineage = cand.lineage_macro_adj - champ.lineage_macro_adj
    all_aggregations_up = d_micro > 1e-9 and d_macro > 1e-9 and d_lineage > 1e-9
    identical = (
        abs(d_micro) < 1e-9
        and abs(d_macro) < 1e-9
        and abs(d_lineage) < 1e-9
        and all(abs(v) < 1e-9 for v in delta_per_ds.values())
    )

    # Promotion decision.
    if identical:
        result = "inconclusive"
        reason = (
            "Gate fired on no edge across all 4 CV families (champion CV byte-"
            "identical) -> no A/B signal on this holdout; keep default-off."
        )
    elif d_micro > NOISE_BAND and no_regression and all_aggregations_up:
        result = "promoted"
        reason = (
            f"micro +{d_micro:.4f} > noise band {NOISE_BAND}, 4/4 non-regression, "
            "micro+macro+lineage-macro all up."
        )
    elif not no_regression or d_micro < -NOISE_BAND:
        result = "rejected"
        reason = (
            f"per-dataset regression ({not no_regression}) or micro {d_micro:+.4f} "
            f"below -{NOISE_BAND}; gate removed real TP edges. Revert / keep off."
        )
    else:
        result = "inconclusive"
        reason = (
            f"micro {d_micro:+.4f} within +/-{NOISE_BAND} noise band "
            f"(no_regression={no_regression}); no promotable signal."
        )

    champ_frozen = sha256(CHAMPION_CONFIG) == CHAMPION_CONFIG_SHA256

    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2895",
        "axis": "post-hoc suspicious-tracking-event review gate (dalloliogm port)",
        "candidate_config": str(CANDIDATE_CONFIG.relative_to(REPO)),
        "effective_config": effective,
        "postprocessing_only_change": postproc_only,
        "champion_cv": cv_result_to_dict(champ),
        "candidate_cv": cv_result_to_dict(cand),
        "champion_per_dataset_adj": champ_per_ds,
        "candidate_per_dataset_adj": cand_per_ds,
        "delta_per_dataset_adj": delta_per_ds,
        "delta_micro_adj": round(d_micro, 4),
        "delta_macro_adj": round(d_macro, 4),
        "delta_lineage_macro_adj": round(d_lineage, 4),
        "no_per_dataset_regression": bool(no_regression),
        "all_aggregations_up": bool(all_aggregations_up),
        "champion_cv_byte_identical": bool(identical),
        "candidate_representativeness": representativeness_report(cand),
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "champion_reference_score": CHAMPION_REFERENCE_SCORE,
        "champion_config_sha256": sha256(CHAMPION_CONFIG),
        "champion_config_byte_frozen": champ_frozen,
        "candidate_config_sha256": sha256(CANDIDATE_CONFIG),
        "result": result,
        "reason": reason,
        "kaggle_submitted": False,
    }
    out = REPO / "experiments/sot2895/ab_suspicious_review.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"[champion]  micro={champ.micro_adj_edge_jaccard:.4f} "
        f"macro={champ.macro_adj_edge_jaccard:.4f} "
        f"lineage_macro={champ.lineage_macro_adj:.4f}",
        flush=True,
    )
    print(
        f"[candidate] micro={cand.micro_adj_edge_jaccard:.4f} "
        f"macro={cand.macro_adj_edge_jaccard:.4f} "
        f"lineage_macro={cand.lineage_macro_adj:.4f}",
        flush=True,
    )
    print(f"[delta] micro={d_micro:+.4f} per_dataset={delta_per_ds}", flush=True)
    print(f"[postproc_only={postproc_only}] [champ_frozen={champ_frozen}]", flush=True)
    print(f"RESULT={result} :: {reason}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
