"""SOT-2902 — champion reproducibility forensics: confirm the public-0.624 restore candidate.

Drives verification END-TO-END from the restore candidate config file
(``champion/candidates/sot2902-restore-0624.json``):

1. **Effective config == champion** — ``champion_params(restore)`` yields detect+link
   knobs byte-identical to the reigning champion (this restore IS the 0.624-generating
   config; it was never lost — the champion has been byte-frozen since it produced 0.624).
2. **Restore CV** — on the SOT-2817 re-anchored full-metric leak-free CV the restore
   candidate reproduces micro-adj 0.6649 == champion_reference (delta 0.0000).
3. **Exec-compat** — the champion kernel that carries this config passes the exec-runtime
   gate (no ``__file__``, undefined cwd).
4. **Fingerprint forensics** — record that the frozen champion kernel sha256 == 48b1eaa2…
   (== the 0.624 submission's artifact fingerprint, ref 55212214), and that e445965…
   (the 0.509 submission, ref 55649179) == sha256('kaggle-notebook:<kernel>@4:submission.csv'),
   the synthetic fallback identity — NOT a content hash. This proves HEAD regenerates 48b1e,
   not e445965, and the '48b1e vs e445965' pair is a fingerprint-scheme flip, not a regression.

Writes ``experiments/sot2902/confirm_restore_0624.json``. No Kaggle submission.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.eval.cv import (
    CHAMPION_REFERENCE_MICRO_ADJ,
    CHAMPION_REFERENCE_SCORE,
    cv_result_to_dict,
    evaluate_cv,
)

REPO = Path(__file__).resolve().parents[2]
RESTORE_CONFIG = REPO / "champion/candidates/sot2902-restore-0624.json"
CHAMPION_CONFIG = REPO / "champion/config.json"
CHAMPION_KERNEL = REPO / "submit/kernel/biohub-claude-champion.py"

CHAMPION_CONFIG_SHA256 = "42064648e612183e761bf9d40b70d3e8a2497453a878f1a44f5b52e410e01bdd"
# The public-0.624 submission's artifact fingerprint (Kaggle ref 55212214, 2026-08-03)
# == sha256(submit/kernel/biohub-claude-champion.py) at the SOT-2369 commit 57bdf4f.
ARTIFACT_0624_SHA256 = "48b1eaa2dfb63c8a31344c8e5695acb8c7d546ba696952ce317ed6471ee3a698"
# The public-0.509 submission's fingerprint (Kaggle ref 55649179, 2026-08-20): the
# SYNTHETIC kaggle-notebook fallback identity, NOT a content hash.
ARTIFACT_0509_SHA256 = "e445965c6cdea076430fc7a0a899aca9c24933877d72998e734dff0f3623da8e"
KERNEL_ID = "sota1111/biohub-claude-champion"
FALLBACK_IDENTITY_VERSION = 4  # registry submit.version at the 2026-08-20 submit


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def main() -> int:
    restore_cfg = json.loads(RESTORE_CONFIG.read_text())
    r_detect, r_link, r_scale = champion_params(restore_cfg)
    c_detect, c_link, c_scale = champion_params(load_champion_config(CHAMPION_CONFIG))

    # 1) effective config identical to champion (this IS the 0.624 config)
    effective_config_identical = (
        r_detect.mad_k == c_detect.mad_k
        and r_detect.threshold_percentile == c_detect.threshold_percentile
        and tuple(r_detect.sigma_zyx) == tuple(c_detect.sigma_zyx)
        and tuple(r_detect.background_sigma_zyx) == tuple(c_detect.background_sigma_zyx)
        and tuple(r_detect.nms_size_zyx) == tuple(c_detect.nms_size_zyx)
        and r_detect.min_threshold == c_detect.min_threshold
        and r_link.max_distance == c_link.max_distance
        and r_link.allow_division == c_link.allow_division
        and r_link.division_distance == c_link.division_distance
        and r_link.min_track_length == c_link.min_track_length
        and tuple(r_scale) == tuple(c_scale)
    )

    # 2) restore-candidate leak-free CV (== the 0.624-config CV)
    restore_result = evaluate_cv(config=restore_cfg)
    restore_cv = cv_result_to_dict(restore_result)
    delta_micro = abs(restore_result.micro_adj_edge_jaccard - CHAMPION_REFERENCE_MICRO_ADJ)
    delta_score = abs(restore_result.score - CHAMPION_REFERENCE_SCORE)
    cv_reproduces_reference = delta_micro <= 1e-4 and delta_score <= 1e-4

    # 3) exec-compat gate (champion kernel that carries this config)
    exec_compat_ok = False
    exec_compat_summary = None
    try:
        sys.path.insert(0, str(REPO / "submit"))
        import exec_compat_gate  # noqa: E402

        exec_compat_summary = exec_compat_gate.run_gate()
        exec_compat_ok = True
    except Exception as exc:  # pragma: no cover - surfaced in the JSON
        exec_compat_summary = {"error": repr(exc)}

    # 4) fingerprint forensics
    champion_kernel_sha256 = sha256_file(CHAMPION_KERNEL)
    champion_config_sha256 = sha256_file(CHAMPION_CONFIG)
    fallback_identity = f"kaggle-notebook:{KERNEL_ID}@{FALLBACK_IDENTITY_VERSION}:submission.csv"
    fallback_identity_sha256 = sha256_bytes(fallback_identity.encode())

    head_regenerates_0624 = champion_kernel_sha256 == ARTIFACT_0624_SHA256
    e445965_is_fallback_identity = fallback_identity_sha256 == ARTIFACT_0509_SHA256

    payload = {
        "recordedAt": datetime.now(timezone.utc).isoformat(),
        "issue": "SOT-2902",
        "restore_candidate": str(RESTORE_CONFIG.relative_to(REPO)),
        "effective_config_identical_to_champion": effective_config_identical,
        "restore_cv": {
            "micro_adj_edge_jaccard": round(restore_result.micro_adj_edge_jaccard, 4),
            "score": round(restore_result.score, 4),
            "division_term": round(restore_result.division_term, 4),
            "delta_micro_vs_reference": round(delta_micro, 6),
            "delta_score_vs_reference": round(delta_score, 6),
            "cv_reproduces_reference": cv_reproduces_reference,
            "reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        },
        "exec_compat_ok": exec_compat_ok,
        "exec_compat_summary": exec_compat_summary,
        "fingerprint_forensics": {
            "champion_kernel_sha256": champion_kernel_sha256,
            "champion_config_sha256": champion_config_sha256,
            "champion_config_sha256_expected": CHAMPION_CONFIG_SHA256,
            "artifact_0624_fingerprint": ARTIFACT_0624_SHA256,
            "artifact_0509_fingerprint": ARTIFACT_0509_SHA256,
            "head_kernel_sha256_equals_0624_artifact": head_regenerates_0624,
            "fallback_identity_string": fallback_identity,
            "fallback_identity_sha256": fallback_identity_sha256,
            "e445965_is_kaggle_notebook_fallback_identity": e445965_is_fallback_identity,
            "note": (
                "48b1e == sha256(champion kernel .py) = real source-file hash reproduced at HEAD; "
                "e445965 == sha256(kaggle-notebook:<kernel>@4:submission.csv) = synthetic fallback "
                "identity (kernel-name@version+output only, never the scored CSV content). The two "
                "are different fingerprint SCHEMES, not two submission artifacts. HEAD regenerates "
                "48b1e (the 0.624 source), never e445965."
            ),
        },
        "per_dataset": restore_cv.get("per_dataset"),
    }

    out = REPO / "experiments/sot2902/confirm_restore_0624.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in (
        "effective_config_identical_to_champion",
        "restore_cv",
        "exec_compat_ok",
    )}, indent=2))
    print(f"head_regenerates_0624={head_regenerates_0624} "
          f"e445965_is_fallback_identity={e445965_is_fallback_identity}")
    print(f"wrote {out.relative_to(REPO)}")

    all_ok = (
        effective_config_identical
        and cv_reproduces_reference
        and exec_compat_ok
        and head_regenerates_0624
        and e445965_is_fallback_identity
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
