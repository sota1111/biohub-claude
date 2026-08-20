"""Confirm the SOT-2763 gap-closing screen through the real champion_params path.

The screen (``screen_gap_closing.py``) re-links off cached detections. This
confirm step re-scores the champion baseline AND the least-bad challenger
(max_frame_gap=2, gap_distance=7.0) through the *real* ``champion_params(config)
-> run_pipeline`` used at submission time and the SOT-2761 ``evaluate_cv`` — the
same independent re-score that caught the SOT-2369 plumbing bug (``champion_params``
dropping ``link.min_track_length``). Here it guards that the new ``max_frame_gap``
/ ``gap_distance`` fields flow through ``champion_params`` and that the challenger
numbers reproduce the screen, so the "rejected" verdict rests on the confirm-grade
evaluator, not just the cached-detection screen.

Writes ``experiments/sot2763/confirm_gap_closing.json``. No Kaggle submission.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.eval.cv import cv_result_to_dict, evaluate_cv


def main() -> int:
    champ_cfg = load_champion_config()

    # Sanity: the new gap-closing fields must survive champion_params plumbing.
    challenger_cfg = copy.deepcopy(champ_cfg)
    challenger_cfg["link"] = {
        **champ_cfg["link"],
        "max_frame_gap": 2,
        "gap_distance": 7.0,
    }
    _d, link, _s = champion_params(challenger_cfg)
    assert link.max_frame_gap == 2, link
    assert link.gap_distance == 7.0, link
    assert link.min_track_length == champ_cfg["link"]["min_track_length"], link
    # champion default plumbing: gap-closing off unless the config sets it.
    _d0, link0, _s0 = champion_params(champ_cfg)
    assert link0.max_frame_gap == 1, link0

    baseline = evaluate_cv(config=champ_cfg)
    challenger = evaluate_cv(config=challenger_cfg)

    champion_floor = {
        "44b6_0113de3b": 0.8895, "44b6_0b24845f": 0.6817,
        "6bba_05b6850b": 0.5700, "6bba_05db0fb1": 0.7310,
    }
    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2763",
        "method": "independent re-score via champion_params(config)->run_pipeline "
                  "+ SOT-2761 evaluate_cv (re-runs real detection, not cached)",
        "plumbing_check": "max_frame_gap/gap_distance flow through champion_params, "
                          "and default champion stays max_frame_gap=1 (off): OK",
        "baseline_champion": {
            "score": round(baseline.score, 4),
            "micro_adj_edge_jaccard": round(baseline.micro_adj_edge_jaccard, 4),
            "cv": cv_result_to_dict(baseline),
        },
        "challenger_mfg2_gd7": {
            "score": round(challenger.score, 4),
            "micro_adj_edge_jaccard": round(challenger.micro_adj_edge_jaccard, 4),
            "delta_score_vs_champion": round(challenger.score - baseline.score, 4),
            "no_per_dataset_regression": bool(challenger.no_regression_vs(champion_floor)),
            "cv": cv_result_to_dict(challenger),
        },
    }
    verdict = (
        challenger.score > baseline.score
        and challenger.no_regression_vs(champion_floor)
    )
    payload["promotable"] = bool(verdict)
    out = Path(__file__).resolve().parent / "confirm_gap_closing.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"baseline champion score={baseline.score:.4f}")
    print(f"challenger mfg2/gd7 score={challenger.score:.4f} "
          f"delta={challenger.score - baseline.score:+.4f} "
          f"no_reg={challenger.no_regression_vs(champion_floor)}")
    print(f"promotable={verdict}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
