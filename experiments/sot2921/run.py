"""SOT-2921: GT-free per-sequence density-regime covariate + leak-free separation.

Builds, for each of the four leak-free CV sequences, an **observable** density /
crowding covariate from the champion detector's point cloud alone
(:func:`biohub_tracking.eval.regime.sequence_covariates`) — no ground truth, no
family label — and audits whether that covariate separates the **dense** 6bba
lineage from the **sparse/clean** 44b6 lineage (the family-mix wall). It then
emits a **regime-stratified** champion CV report (per-family + per-regime
micro_adj / micro_raw), using the *observed* covariate — not the family label —
to assign each sequence's regime, so the report is itself leak-free.

Detection is byte-identical to the champion (this issue changes no detect/link
logic); detections are computed ONCE per family and reused for both the covariate
and the champion CV, which reproduces the registry micro_adj (self-check). No
Kaggle submission; mutates no champion state.

Writes ``experiments/sot2921/covariate_separation.json``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.detect import DEFAULT_SCALE, detect_volume_series
from biohub_tracking.eval.cv import (
    CHAMPION_REFERENCE_MICRO_ADJ,
    CV_HOLDOUT,
    aggregate,
    cv_result_to_dict,
    score_family,
)
from biohub_tracking.eval.regime import (
    assign_regime,
    regime_stratified_cv,
    separation_report,
    sequence_covariates,
)
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff
from biohub_tracking.link import link_centroids
from biohub_tracking.pipeline import _open_image_array

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "covariate_separation.json"
PRIMARY_KEY = "median_knn_um"  # spacing: small ⇒ dense, large ⇒ sparse


def _family_prefix(name: str) -> str:
    """Lineage prefix (``6bba`` / ``44b6``) — used ONLY to score separation."""
    return name.split("_", 1)[0]


def main() -> int:
    champ_cfg = load_champion_config()
    detect_params, champ_link, _cfg_scale = champion_params(champ_cfg)

    # Detect once per family. GT-free: only the test *.zarr is read here; the
    # *.geff is loaded solely to SCORE the champion CV afterwards.
    detections_by_fam: dict[str, dict] = {}
    covariates_by_seq: dict[str, dict] = {}
    rows = []
    for fam in CV_HOLDOUT:
        arr = _open_image_array(REPO / fam.image)
        det = detect_volume_series(arr, detect_params)
        detections_by_fam[fam.name] = det

        # --- observable covariate: detection cloud only, fixed acquisition scale.
        covariates_by_seq[fam.name] = sequence_covariates(det, scale=DEFAULT_SCALE)

        # --- champion CV row (GT used for scoring only; never for the covariate).
        geff = REPO / fam.geff
        pred = link_centroids(det, scale=geff_scale(geff), params=champ_link)
        rows.append(
            score_family(
                fam,
                pred,
                load_geff(geff),
                geff_estimated_num_nodes(geff),
                scale=geff_scale(geff),
            )
        )
        print(
            f"[{fam.name}] {PRIMARY_KEY}="
            f"{covariates_by_seq[fam.name][PRIMARY_KEY]:.3f}um  "
            f"density={covariates_by_seq[fam.name]['density_cells_per_1e6_um3']:.4f}  "
            f"crowding={covariates_by_seq[fam.name]['local_crowding']:.2f}  "
            f"cells/frame={covariates_by_seq[fam.name]['median_cells_per_frame']:.0f}",
            flush=True,
        )

    champ_cv = aggregate(rows)

    # --- leak-free separation audit on the primary covariate.
    separation = separation_report(
        covariates_by_seq, _family_prefix, key=PRIMARY_KEY, dense_is_low=True
    )
    # Auxiliary covariates' separation (count-type: larger ⇒ dense).
    aux_separation = {
        "density_cells_per_1e6_um3": separation_report(
            covariates_by_seq,
            _family_prefix,
            key="density_cells_per_1e6_um3",
            dense_is_low=False,
        ),
        "local_crowding": separation_report(
            covariates_by_seq, _family_prefix, key="local_crowding", dense_is_low=False
        ),
    }

    # --- regime label per family from the OBSERVED covariate (not the label).
    threshold = separation.get("threshold")
    if threshold is None:  # degenerate: fall back to per-family value ranking
        regime_by_family = {
            name: "unknown" for name in covariates_by_seq
        }
    else:
        regime_by_family = {
            name: assign_regime(
                covariates_by_seq[name][PRIMARY_KEY], threshold, dense_is_low=True
            )
            for name in covariates_by_seq
        }
    regime_report = regime_stratified_cv(rows, regime_by_family)

    champ_micro = champ_cv.micro_adj_edge_jaccard
    payload = {
        "issue": "SOT-2921",
        "recordedAt": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "champion": champ_cfg.get("name"),
        "acquisition_scale_um_per_voxel": list(DEFAULT_SCALE),
        "primary_covariate": PRIMARY_KEY,
        "covariates": covariates_by_seq,
        "separation_primary": separation,
        "separation_auxiliary": aux_separation,
        "regime_labels_from_observable_covariate": regime_by_family,
        "regime_stratified_cv": regime_report,
        "champion_cv": cv_result_to_dict(champ_cv),
        "champion_cv_micro_adj": round(champ_micro, 4),
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "champion_cv_reproduced": abs(champ_micro - 0.6760) < 1e-3,
        "leakage_audit": {
            "covariate_uses_gt": False,
            "covariate_test_computable": True,
            "covariate_uses_family_label": False,
            "note": (
                "sequence_covariates() consumes only the detection point cloud + a "
                "fixed acquisition scale constant; the family prefix is used solely "
                "to score separation and to VERIFY the observable regime labels, "
                "never to compute the covariate or the regime label."
            ),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\n=== SEPARATION (primary: {PRIMARY_KEY}) ===", flush=True)
    print(
        f"separable={separation['separable']} margin={separation['margin']:.3f}um "
        f"threshold={separation['threshold']:.3f}um "
        f"dense_group={separation['dense_group']} "
        f"all_correct={separation['all_correct']}",
        flush=True,
    )
    print("regime labels (observable):", regime_by_family, flush=True)
    for regime, blk in regime_report["per_regime"].items():
        print(
            f"  regime={regime}: micro_adj={blk['micro_adj_edge_jaccard']} "
            f"micro_raw={blk['micro_edge_jaccard']} "
            f"weight_share={blk['weight_share']} families={blk['families']}",
            flush=True,
        )
    print(
        f"champion CV micro_adj={champ_micro:.4f} "
        f"(reproduced 0.6760: {payload['champion_cv_reproduced']})",
        flush=True,
    )
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
