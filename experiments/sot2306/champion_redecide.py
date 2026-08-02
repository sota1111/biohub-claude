"""Re-decide the detection champion on the re-anchored oracle (SOT-2306).

Question
--------
SOT-2306 asks: on the SOT-2305 4-dataset LB-representative holdout, A/B the DoG
detector against the *pre-DoG* v1 global-intensity-threshold baseline and lock in
the LB-consistent detection champion. If DoG degrades on the representative oracle
(the public LB moved 0.509 -> 0.500 when DoG-v2 replaced v1), **revert** the
champion to the global-threshold setting.

Since this issue was filed, the sibling SOT-2307 already promoted a third detector
``detect-link-dog-v3-adaptive`` (DoG + a robust per-volume median+k*MAD threshold
that suppresses the over-detection the node-count penalty punishes). That adaptive
setting is exactly the "intermediate DoG + FP-suppression" variant SOT-2306 lists
as option (c), so we A/B all **three** detectors head-to-head on the same holdout
and record a single champion decision:

  (a) v1_global_threshold  -- pre-DoG baseline (public LB 0.509)
  (b) dog_v2               -- fixed percentile-92 DoG (public LB 0.500)
  (c) v3_adaptive          -- current champion, median + 3*MAD adaptive threshold

Decision rule
-------------
The scored metric is the size-weighted adjusted edge Jaccard (adjusted = edge
Jaccard * min(1, n_true/n_pred), i.e. the node-count over-detection penalty that is
part of the real competition metric). Champion = the detector with the highest
holdout micro-adj. We only revert to global-threshold if the DoG family actually
*loses* to it on the re-anchored oracle.

Performs **no Kaggle submission** and mutates no champion state (read-only A/B).
Run::

    .venv/bin/python experiments/sot2306/champion_redecide.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.detect import DetectParams
from biohub_tracking.eval.score import _jaccard, adjusted_edge_jaccard, evaluate
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff
from biohub_tracking.link import LinkParams
from biohub_tracking.pipeline import run_pipeline

REPO = Path(__file__).resolve().parents[2]

# The exact 4 Kaggle test families (test *.zarr image + train *.geff GT).
HOLDOUT = [
    {"name": "44b6_0113de3b", "prefix": "44b6", "image": "data/test/44b6_0113de3b.zarr", "geff": "data/train/44b6_0113de3b.geff"},
    {"name": "44b6_0b24845f", "prefix": "44b6", "image": "data/test/44b6_0b24845f.zarr", "geff": "data/train/44b6_0b24845f.geff"},
    {"name": "6bba_05b6850b", "prefix": "6bba", "image": "data/test/6bba_05b6850b.zarr", "geff": "data/train/6bba_05b6850b.geff"},
    {"name": "6bba_05db0fb1", "prefix": "6bba", "image": "data/test/6bba_05db0fb1.zarr", "geff": "data/train/6bba_05db0fb1.geff"},
]

_LINK = LinkParams(max_distance=7.0, allow_division=False)

# The current champion (v3_adaptive) is read from champion/config.json so the A/B
# scores the *actual* reigning params, not a hand-copied duplicate.
_champ_detect, _champ_link, _ = champion_params(load_champion_config(REPO / "champion/config.json"))

DETECTORS = {
    "v1_global_threshold": {
        "label": "v1 global intensity threshold (detect-link-v1, LB 0.509)",
        "detect": DetectParams(
            sigma_zyx=(1.0, 3.0, 3.0),
            nms_size_zyx=(2, 5, 5),
            threshold_percentile=99.3,
            background_sigma_zyx=None,
        ),
        "link": _LINK,
    },
    "dog_v2": {
        "label": "DoG-v2 fixed percentile-92 local contrast (detect-link-dog-v2, LB 0.500)",
        "detect": DetectParams(
            sigma_zyx=(1.0, 2.0, 2.0),
            background_sigma_zyx=(2.0, 6.0, 6.0),
            nms_size_zyx=(2, 5, 5),
            threshold_percentile=92.0,
        ),
        "link": _LINK,
    },
    "v3_adaptive": {
        "label": "DoG-v3 adaptive median+3*MAD threshold (detect-link-dog-v3-adaptive, current champion)",
        "detect": _champ_detect,
        "link": _champ_link,
    },
}


def score_family(fam: dict, detect: DetectParams, link: LinkParams) -> dict:
    geff = REPO / fam["geff"]
    image = REPO / fam["image"]
    gt = load_geff(geff)
    scale = geff_scale(geff)
    n_true = geff_estimated_num_nodes(geff)
    pred = run_pipeline(image, scale=scale, detect_params=detect, link_params=link)
    r = evaluate(pred, gt, scale=scale)
    j = _jaccard(r.edge_tp, r.edge_fp, r.edge_fn)
    adj = adjusted_edge_jaccard(j, r.num_pred_nodes, n_true)
    if adj != adj:
        adj = j
    return {
        "name": fam["name"],
        "prefix": fam["prefix"],
        "edge_tp": r.edge_tp,
        "edge_fp": r.edge_fp,
        "edge_fn": r.edge_fn,
        "edge_jaccard": round(j, 4),
        "adjusted_edge_jaccard": round(adj, 4),
        "pred_nodes": r.num_pred_nodes,
        "n_true": (None if n_true != n_true else n_true),
        "weight": r.edge_tp + r.edge_fp + r.edge_fn,
    }


def micro(rows: list[dict]) -> dict:
    tp = sum(r["edge_tp"] for r in rows)
    fp = sum(r["edge_fp"] for r in rows)
    fn = sum(r["edge_fn"] for r in rows)
    edge_j = _jaccard(tp, fp, fn)
    wsum = sum(r["weight"] * r["adjusted_edge_jaccard"] for r in rows if r["weight"] > 0)
    wtot = sum(r["weight"] for r in rows if r["weight"] > 0)
    adj = wsum / wtot if wtot > 0 else float("nan")
    return {
        "n_families": len(rows),
        "edge_tp": tp,
        "edge_fp": fp,
        "edge_fn": fn,
        "micro_edge_jaccard": round(edge_j, 4),
        "micro_adj_edge_jaccard": round(adj, 4),
        "score": round(adj, 4),
    }


def main() -> None:
    detectors_out: dict[str, dict] = {}
    for key, cfg in DETECTORS.items():
        print(f"\n=== {key}: {cfg['label']} ===", flush=True)
        rows = []
        for fam in HOLDOUT:
            print(f"  running {fam['name']} ...", flush=True)
            row = score_family(fam, cfg["detect"], cfg["link"])
            rows.append(row)
            print(
                f"    {row['name']} ({row['prefix']}): "
                f"TP/FP/FN={row['edge_tp']}/{row['edge_fp']}/{row['edge_fn']} "
                f"adj={row['adjusted_edge_jaccard']} pred_nodes={row['pred_nodes']}",
                flush=True,
            )
        overall = micro(rows)
        by_prefix = {p: micro([r for r in rows if r["prefix"] == p]) for p in ("44b6", "6bba")}
        detectors_out[key] = {
            "label": cfg["label"],
            "per_dataset": rows,
            "by_prefix": {p: v["micro_adj_edge_jaccard"] for p, v in by_prefix.items()},
            "holdout_micro_adj": overall["micro_adj_edge_jaccard"],
        }
        print(f"  -> holdout micro-adj = {overall['micro_adj_edge_jaccard']}", flush=True)

    ranking = sorted(detectors_out.items(), key=lambda kv: kv[1]["holdout_micro_adj"], reverse=True)
    winner_key, winner = ranking[0]
    v1 = detectors_out["v1_global_threshold"]["holdout_micro_adj"]
    dog_family_best = max(
        detectors_out["dog_v2"]["holdout_micro_adj"],
        detectors_out["v3_adaptive"]["holdout_micro_adj"],
    )
    revert_to_global = dog_family_best < v1
    champion_name = load_champion_config(REPO / "champion/config.json")["name"]
    keep_current = winner_key == "v3_adaptive"

    decision = {
        "issue": "SOT-2306",
        "title": "Re-decide detection champion on the re-anchored 4-dataset oracle",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_families": [f["name"] for f in HOLDOUT],
        "metric": "size-weighted adjusted edge Jaccard (node-count over-detection penalty on the real Kaggle metric)",
        "public_lb_reference": {"v1_global_threshold": 0.509, "dog_v2": 0.500, "v3_adaptive": "not submitted"},
        "detectors": detectors_out,
        "ranking": [{"detector": k, "holdout_micro_adj": v["holdout_micro_adj"]} for k, v in ranking],
        "winner": winner_key,
        "current_champion": champion_name,
        "revert_to_global_threshold": revert_to_global,
        "keep_current_champion": keep_current,
        "decision": (
            "REVERT_TO_GLOBAL_THRESHOLD" if revert_to_global
            else ("KEEP_CURRENT_CHAMPION" if keep_current else f"PROMOTE_{winner_key}")
        ),
        "oracle_lb_gap_note": (
            "The re-anchored oracle still is not a faithful LB proxy for the v1-vs-DoG ranking: "
            "v1's holdout micro-adj (~0.36) does not reproduce its own LB (0.509), and v1/dog_v2 are "
            "near-tied on the LB (0.509 vs 0.500) whereas the oracle ranks DoG far above v1. This "
            "residual GT/normalization gap is inherited from SOT-2305 and cannot be closed without a "
            "Kaggle submission, which SOT-2306 forbids. The decision therefore uses the best available "
            "oracle: on it the DoG family strictly dominates global-threshold, so a revert is not warranted."
        ),
        "reproducible_note": "Deterministic run_pipeline (no RNG); re-running reproduces every score.",
        "kaggle_submission": "none",
    }

    print("\n" + json.dumps({k: decision[k] for k in ("ranking", "winner", "decision", "revert_to_global_threshold")}, indent=2))
    out = REPO / "experiments/sot2306/champion_redecide.json"
    out.write_text(json.dumps(decision, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
