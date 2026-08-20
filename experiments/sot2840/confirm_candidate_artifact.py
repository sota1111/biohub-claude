"""Confirm the SOT-2840 global-MCF theta=6.5/window=2 LB-probe candidate artifact.

This drives the verification END-TO-END from the *candidate config file*
(``champion/candidates/sot2840-globalmcf-theta6p5-window2.json``) — not a
hand-built LinkParams — so it proves the exact artifact the parent resume run
will submit reproduces SOT-2830's non-regressing CV:

1. **Effective config** — ``champion_params(candidate)`` yields the global-MCF
   linking knobs (global_window=2, birth=death=3.25) while every detection knob
   stays identical to the champion.
2. **Candidate CV** — on the SOT-2817 re-anchored full-metric leak-free CV the
   candidate reproduces micro 0.6671, 4/4 per-dataset NON-REGRESSION vs the
   champion, micro/macro/lineage-macro ALL up, and re-flags family_mix_sensitive.
3. **Live champion byte-frozen** — champion/config.json sha256 is unchanged
   (42064648…) and the champion still reproduces 0.6649 (from the sibling
   ``champion_cv_repro.json``).
4. **Fingerprints** — the candidate config + candidate kernel sha256 differ from
   the champion (the last-submitted artifact).

Writes ``experiments/sot2840/confirm_candidate_artifact.json``. No Kaggle submission.
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
CANDIDATE_CONFIG = REPO / "champion/candidates/sot2840-globalmcf-theta6p5-window2.json"
CHAMPION_CONFIG = REPO / "champion/config.json"
CANDIDATE_KERNEL = REPO / "submit/kernel-candidate/biohub-claude-candidate.py"
CHAMPION_KERNEL = REPO / "submit/kernel/biohub-claude-champion.py"
CHAMPION_CONFIG_SHA256 = (
    "42064648e612183e761bf9d40b70d3e8a2497453a878f1a44f5b52e410e01bdd"
)

# SOT-2830 confirmed champion per-dataset adjusted-edge Jaccard (the non-regression
# floor) and the aggregation baseline the candidate must beat on all three axes.
CHAMPION_PER_DATASET_ADJ = {
    "44b6_0113de3b": 0.8895,
    "44b6_0b24845f": 0.6817,
    "6bba_05b6850b": 0.5700,
    "6bba_05db0fb1": 0.7310,
}
CHAMPION_MICRO = 0.6649
CHAMPION_MACRO = 0.7180
CHAMPION_LINEAGE_MACRO = 0.7216
EXPECTED_CANDIDATE_SCORE = 0.6671


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    candidate_cfg = json.loads(CANDIDATE_CONFIG.read_text())
    _detect, link, _scale = champion_params(candidate_cfg)
    champ_detect, champ_link, _ = champion_params(load_champion_config(CHAMPION_CONFIG))

    effective = {
        "global_window": link.global_window,
        "birth_cost": link.birth_cost,
        "death_cost": link.death_cost,
        "max_distance": link.max_distance,
        "min_track_length": link.min_track_length,
        "allow_division": link.allow_division,
        "mad_k": _detect.mad_k,
        "threshold_percentile": _detect.threshold_percentile,
    }
    effective_config_ok = (
        link.global_window == 2
        and abs(link.birth_cost - 3.25) < 1e-12
        and abs(link.death_cost - 3.25) < 1e-12
        # detection knobs identical to champion (linking-only change)
        and _detect.mad_k == champ_detect.mad_k
        and _detect.threshold_percentile == champ_detect.threshold_percentile
        and link.max_distance == champ_link.max_distance
        and link.min_track_length == champ_link.min_track_length
    )
    print(f"[effective] {effective} ok={effective_config_ok}", flush=True)

    res = evaluate_cv(config=candidate_cfg)
    rep = representativeness_report(res)
    per_ds = {r.name: round(r.adj_edge_jaccard, 4) for r in res.per_dataset}

    no_regression = res.no_regression_vs(CHAMPION_PER_DATASET_ADJ)
    score_ok = abs(res.score - EXPECTED_CANDIDATE_SCORE) < 1e-4
    all_aggregations_up = (
        res.micro_adj_edge_jaccard > CHAMPION_MICRO + 1e-9
        and res.macro_adj_edge_jaccard > CHAMPION_MACRO + 1e-9
        and res.lineage_macro_adj > CHAMPION_LINEAGE_MACRO + 1e-9
    )
    family_mix_sensitive = bool(rep.get("family_mix_sensitive"))

    champ_frozen = sha256(CHAMPION_CONFIG) == CHAMPION_CONFIG_SHA256
    cand_cfg_sha = sha256(CANDIDATE_CONFIG)
    cand_kernel_sha = sha256(CANDIDATE_KERNEL) if CANDIDATE_KERNEL.exists() else None
    champ_kernel_sha = sha256(CHAMPION_KERNEL) if CHAMPION_KERNEL.exists() else None
    fingerprint_new = (
        cand_cfg_sha != CHAMPION_CONFIG_SHA256
        and cand_kernel_sha is not None
        and cand_kernel_sha != champ_kernel_sha
    )

    print(
        f"[candidate] score={res.score:.4f} micro={res.micro_adj_edge_jaccard:.4f} "
        f"macro={res.macro_adj_edge_jaccard:.4f} lineage_macro={res.lineage_macro_adj:.4f} "
        f"no_reg={no_regression} all_up={all_aggregations_up} "
        f"family_mix_sensitive={family_mix_sensitive}",
        flush=True,
    )
    print(f"[per_dataset] {per_ds}", flush=True)

    passed = bool(
        effective_config_ok
        and score_ok
        and no_regression
        and all_aggregations_up
        and family_mix_sensitive
        and champ_frozen
        and fingerprint_new
    )

    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2840",
        "candidate_config": str(CANDIDATE_CONFIG.relative_to(REPO)),
        "candidate_kernel": str(CANDIDATE_KERNEL.relative_to(REPO)),
        "effective_config": effective,
        "effective_config_ok": effective_config_ok,
        "candidate_cv": cv_result_to_dict(res),
        "expected_score": EXPECTED_CANDIDATE_SCORE,
        "score_reproduced": score_ok,
        "no_per_dataset_regression": bool(no_regression),
        "all_aggregations_up": bool(all_aggregations_up),
        "family_mix_sensitive": family_mix_sensitive,
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "champion_reference_score": CHAMPION_REFERENCE_SCORE,
        "champion_config_sha256": sha256(CHAMPION_CONFIG),
        "champion_config_byte_frozen": champ_frozen,
        "candidate_config_sha256": cand_cfg_sha,
        "candidate_kernel_sha256": cand_kernel_sha,
        "champion_kernel_sha256": champ_kernel_sha,
        "fingerprint_new_vs_last_submitted": fingerprint_new,
        "per_dataset_adj": per_ds,
        "kaggle_submitted": False,
        "passed": passed,
        "note": (
            "Candidate artifact for the parent resume run's reserve-slot LB probe. "
            "Live champion pointer (champion/config.json sha256 42064648...) stays "
            "BYTE-FROZEN; the candidate is family_mix_sensitive=True so no pointer "
            "flip without LB confirmation (SOT-2817 guard / SOT-2816 hazard)."
        ),
    }
    out = REPO / "experiments/sot2840/confirm_candidate_artifact.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"PASSED={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
