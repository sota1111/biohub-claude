"""Confirm the SOT-2775 watershed screen through the real champion_params path.

The screen (``screen_watershed.py``) already scores through the real
``detect_volume_series`` + ``link_centroids`` + ``score_family`` (not a cached
shortcut), so its per-dataset A/B is confirm-grade. This step adds the two guards
the sibling confirms exist for:

1. **Plumbing.** The new ``detect.watershed`` config field must flow through
   ``champion_params`` (the submission-time path), and the reigning champion
   config must still resolve to ``watershed=None`` — i.e. the champion detector is
   byte-unchanged (this run promotes nothing). Data-free, instant.
2. **champion_params↔screen agreement.** Re-score the champion baseline through
   ``evaluate_cv`` (must reproduce the registry **0.6649**, proving the champion is
   untouched), and re-score the rejected challenger through the *real*
   ``champion_params(config)->run_pipeline`` on one fast family
   (``44b6_0113de3b``) — it must reproduce the screen's ``0.9105``, proving the
   ``champion_params`` watershed wiring produces the same detector the screen used.

The dense ``6bba`` challenger families are intentionally NOT re-run through
``run_pipeline`` here: the watershed detector's per-component grey reconstruction
is pathologically slow on the dense families' large low-contrast foreground
components (itself corroborating over-segmentation), the screen already recorded
their real-link A/B, and nothing is promoted, so a ~20-min re-score would add no
new evidence. Writes ``experiments/sot2775/confirm_watershed.json``. No Kaggle
submission.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.eval.cv import CV_HOLDOUT, cv_result_to_dict, evaluate_cv

REPO = Path(__file__).resolve().parents[2]

WATERSHED = ("hmaxima", 3.0, 8.0, 4.0)  # the screen's surgical best-case challenger
FAST_FAMILY = "44b6_0113de3b"  # a fast NMS-count family to guard the wiring
SCREEN_CHALLENGER_ADJ = 0.9105  # ws_h3_ms8_sd4 on 44b6_0113de3b, from the screen


def main() -> int:
    champ_cfg = load_champion_config()

    # (1) Plumbing: watershed survives champion_params; champion default stays None.
    challenger_cfg = copy.deepcopy(champ_cfg)
    challenger_cfg["detect"] = {**champ_cfg["detect"], "watershed": list(WATERSHED)}
    det_c, _l, _s = champion_params(challenger_cfg)
    assert det_c.watershed == WATERSHED, det_c.watershed
    det_0, _l0, _s0 = champion_params(champ_cfg)
    assert det_0.watershed is None, det_0.watershed

    # (2a) Champion untouched: evaluate_cv reproduces the registry 0.6649.
    baseline = evaluate_cv(config=champ_cfg)

    # (2b) champion_params↔screen agreement on one fast family.
    fam = next(f for f in CV_HOLDOUT if f.name == FAST_FAMILY)
    challenger_fast = evaluate_cv(config=challenger_cfg, families=(fam,))
    fast_adj = challenger_fast.per_dataset[0].adj_edge_jaccard

    plumbing_ok = True
    baseline_ok = abs(baseline.micro_adj_edge_jaccard - 0.6649) < 5e-4
    wiring_ok = abs(fast_adj - SCREEN_CHALLENGER_ADJ) < 5e-4

    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2775",
        "verdict": "REJECT",
        "watershed_challenger": list(WATERSHED),
        "method": "champion_params plumbing guard + evaluate_cv re-score "
        "(baseline all-4 families; challenger 1 fast family to guard the wiring)",
        "plumbing_check": {
            "watershed_flows_through_champion_params": plumbing_ok,
            "champion_default_watershed_is_none": det_0.watershed is None,
        },
        "baseline_champion_micro_adj": round(baseline.micro_adj_edge_jaccard, 4),
        "baseline_reproduces_registry_0_6649": baseline_ok,
        "challenger_fast_family": FAST_FAMILY,
        "challenger_fast_adj": round(fast_adj, 4),
        "challenger_wiring_matches_screen_0_9105": wiring_ok,
        "baseline_champion_full": cv_result_to_dict(baseline),
        "note": "Full 4-family challenger A/B is in screen_watershed.json (real "
        "detect+link+score). Verdict REJECT: net micro-adj 0.6649->0.5869 and "
        "per-dataset no-regression FALSE (44b6_0b24845f 0.6817->0.1079, "
        "6bba_05db0fb1 0.7310->0.5974). Champion detect-link-dog-v4-shorttrack "
        "MAINTAINED; watershed kept as a default-off DetectParams knob.",
    }
    out = REPO / "experiments/sot2775/confirm_watershed.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"baseline micro-adj={baseline.micro_adj_edge_jaccard:.4f} (reproduces 0.6649: {baseline_ok})")
    print(f"challenger[{FAST_FAMILY}] adj={fast_adj:.4f} (matches screen 0.9105: {wiring_ok})")
    print(f"plumbing: watershed through champion_params OK; champion default None OK")
    print(f"VERDICT: REJECT  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
