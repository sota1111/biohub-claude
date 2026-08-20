"""SOT-2816 oracle audit — re-score the champion lineage on the ONE leak-free CV.

Runs every historical champion config (v1 / dog-v2 / dog-v3-adaptive /
v4-shorttrack) through the identical `biohub_tracking.eval.cv` harness, so the
CV series is produced by ONE evaluator (no per-experiment re-implementation that
would let the oracle drift). Also quantifies the division-Jaccard CV headroom:
the holdout GT division events, the 0.1-term ceiling, and its FP sensitivity.

Read-only w.r.t. champion state: never writes champion/config.json or
registry.json. Emits a single JSON report for the audit writeup.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from biohub_tracking.champion import load_champion_config
from biohub_tracking.eval.cv import CV_HOLDOUT, cv_result_to_dict, evaluate_cv
from biohub_tracking.eval.division_metric import extract_divisions
from biohub_tracking.io import load_geff

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# Lineage in chronological champion order. v4 = the reigning champion/config.json.
LINEAGE = [
    ("detect-link-v1", HERE / "config_v1.json"),
    ("detect-link-dog-v2", HERE / "config_dog_v2.json"),
    ("detect-link-dog-v3-adaptive", HERE / "config_dog_v3_adaptive.json"),
    ("detect-link-dog-v4-shorttrack", None),  # None -> reigning champion
]


def _rescore():
    out = []
    for name, cfg_path in LINEAGE:
        cfg = load_champion_config() if cfg_path is None else json.loads(cfg_path.read_text())
        res = evaluate_cv(config=cfg, repo_root=REPO)
        d = cv_result_to_dict(res)
        d["name"] = name
        d["config_source"] = "champion/config.json" if cfg_path is None else cfg_path.name
        out.append(d)
        print(f"[{name}] micro_adj={d['micro_adj_edge_jaccard']} "
              f"div_j={d['division_jaccard']} score={d['score']}")
    return out


def _division_headroom():
    """Count GT division events per holdout family and the 0.1-term ceiling."""
    per_family = {}
    total = 0
    for fam in CV_HOLDOUT:
        gt = load_geff(REPO / fam.geff)
        divs = extract_divisions(gt)
        n = len(divs)
        per_family[fam.name] = n
        total += n
    # Champion forfeits the whole 0.1 term (allow_division=false -> 0 pred forks):
    # div_tp=0, div_fp=0, div_fn=total -> division_jaccard = 0 -> +0.1*0 = 0.
    # Ceiling if ALL recovered with no FP: 3/3 = 1.0 -> +0.1.
    # FP sensitivity: k spurious forks -> J = total/(total+k).
    ceiling = 0.1 * (total / total) if total else 0.0
    fp_sensitivity = {
        f"fp={k}": {
            "division_jaccard": round(total / (total + k), 4) if total else None,
            "score_contribution": round(0.1 * (total / (total + k)), 4) if total else 0.0,
        }
        for k in range(0, 4)
    }
    # Also: recovering r of the events with 0 FP.
    recall_sweep = {
        f"tp={r}/{total}": {
            "division_jaccard": round(r / total, 4) if total else None,
            "score_contribution": round(0.1 * (r / total), 4) if total else 0.0,
        }
        for r in range(0, total + 1)
    }
    return {
        "gt_division_events_per_family": per_family,
        "gt_division_events_total": total,
        "score_term_ceiling_full_recall_no_fp": round(ceiling, 4),
        "champion_forfeit_note": (
            "champion allow_division=false emits 0 forks -> division_tp=fp=0, "
            f"division_fn={total} -> division_jaccard=0 -> +0.1*0 contributes 0"
        ),
        "fp_sensitivity_from_full_recall": fp_sensitivity,
        "recall_sweep_no_fp": recall_sweep,
    }


def main():
    payload = {
        "issue": "SOT-2816",
        "harness": "biohub_tracking.eval.cv.evaluate_cv (single leak-free 4-family CV)",
        "lineage_cv": _rescore(),
        "division_headroom": _division_headroom(),
    }
    out = HERE / "lineage_cv_rescore.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
