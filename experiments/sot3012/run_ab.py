"""SOT-3012 — isolate the LEARNED-DETECTION lever: feed the official royerlab
released-weights DETECTION SUBSTRATE (TemporalUNet3D detection map -> local-max
-> centres) into the CHAMPION's CLASSICAL ARGUS motion-model LAP linking, byte
据置き, and A/B it against the pure-classical champion (micro_adj 0.6760).

Role A' *detection-only* hybrid (SOT-3010 cycle-3 direction 2, 申し送り軸#3). It
is structurally independent of the wholesale learned pipeline (SOT-3011, which
adopted learned detection AND learned linking together and REJECTED 4/4 on the
sparse 44b6 families: greedy micro 0.7705 / ILP 0.8081 but 44b6 regressed). This
run holds the LINKING at champion and swaps ONLY detection, so it disentangles:
is the big 6bba gain (and the 44b6 loss) a DETECTION-substrate effect or a
LINKING effect?

Both arms are scored by the SAME leak-free harness
(``biohub_tracking.eval.cv``); the learned arm's centres are linked by the
SAME ``link_centroids`` + champion ``LinkParams`` (motion_model_link=true,
motion_smooth_sigma=15.0, motion_gain=1.0, min_track_length=4) the champion
uses, so the A/B isolates detection. Learned-linking axes are NOT reintroduced
(SOT-2841/2870/2994 already saturated them 3×).

LEAK CAVEAT (recorded honestly): the released split_0 weights were trained on
the organiser's labelled train videos, which INCLUDE these four CV families. So
the learned-detection arm's CV is an *optimistic upper bound* (train-contaminated
detector), not a leak-free estimate. That only strengthens a REJECT and heavily
caveats any PASS — same caveat as SOT-3011.

No Kaggle submission (child issue; the parent decides submission).
Reuses SOT-3011's fetched infra: the royerlab clone + released weights under
``experiments/sot3011/_ext``. Runs in the repo ``.venv`` (torch+cuda).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Reuse SOT-3011's fetched external infra (clone + released weights) — do NOT
# re-download. The learned DETECTION substrate is identical; only the linking
# stage differs (classical champion here vs learned there).
EXT = REPO / "experiments/sot3011/_ext/royerlab"
ART = REPO / "experiments/sot3011/_ext/artifacts/cellmot-baseline-artifacts"
WEIGHTS = ART / "weights/unet_transformer/split_0/edge_predictor_best.pth"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(EXT / "src"))
sys.path.insert(0, str(EXT / "scripts"))

import numpy as np  # noqa: E402

from biohub_tracking.champion import champion_params, load_champion_config  # noqa: E402
from biohub_tracking.eval.cv import (  # noqa: E402
    CV_HOLDOUT,
    aggregate,
    cv_result_to_dict,
    evaluate_cv,
    score_family,
)
from biohub_tracking.io import (  # noqa: E402
    geff_estimated_num_nodes,
    geff_scale,
    load_geff,
)
from biohub_tracking.link import link_centroids  # noqa: E402

NOISE = 0.003


def _log(m: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def _detections_by_t(coords: np.ndarray) -> dict[int, np.ndarray]:
    """Group official detection ``coords[N,4]=(t,z,y,x)`` (original-resolution
    voxel space) into the ``{t: (M,3) zyx}`` mapping ``link_centroids`` expects.

    Node ordering within a timepoint is preserved (row order), matching the
    deterministic id assignment ``link_centroids`` performs."""
    by_t: dict[int, list] = defaultdict(list)
    for t, z, y, x in coords:
        by_t[int(t)].append((float(z), float(y), float(x)))
    return {t: np.asarray(v, dtype=float) for t, v in sorted(by_t.items())}


def learned_detect_champion_link_cv(model_bundle, det_threshold, device,
                                    max_frames, champ_link) -> tuple[dict, list]:
    """Learned detection centres -> CHAMPION classical linking, over the CV holdout.

    Returns ``(aggregated_cv_dict, per_family_internal_rows)``."""
    import predict_unet_transformer as P

    model, window_size, downsample = model_bundle
    cfg = P.PredictConfig(det_threshold=det_threshold, use_ilp=False)
    rows = []
    internals = []
    for fam in CV_HOLDOUT:
        geff = REPO / fam.geff
        image = REPO / fam.image
        scale = geff_scale(geff)
        gt = load_geff(geff)
        n_true = geff_estimated_num_nodes(geff)

        t0 = time.time()
        coords, _learned_edges = P.predict_video(
            model, image, device, cfg=cfg, window_size=window_size,
            max_frames=max_frames, downsample=downsample,
        )
        n_det = len(coords)
        detections = _detections_by_t(coords)
        # SAME linking stage the champion uses: link_centroids + champion
        # LinkParams (motion-model LAP, min_track_length=4). scale == geff_scale,
        # identical to evaluate_cv's champion arm. No descriptors => distance-only
        # motion champion (byte-identical linking; learned-linking NOT reintroduced).
        pred = link_centroids(detections, scale=scale, params=champ_link)
        row = score_family(fam, pred, gt, n_true, scale=scale)
        rows.append(row)
        internals.append({
            "name": fam.name,
            "n_det_learned": int(n_det),
            "pred_nodes_after_link": int(pred.num_nodes),
            "pred_edges": int(pred.num_edges),
            "edge_tp": int(row.edge_tp),
            "edge_fp": int(row.edge_fp),
            "edge_fn": int(row.edge_fn),
            "adjusted_edge_jaccard": round(float(row.adj_edge_jaccard), 4),
        })
        _log(f"  [learned-det+champ-link|thr={det_threshold}|{fam.name}] "
             f"det={n_det} kept_nodes={pred.num_nodes} edges={pred.num_edges} "
             f"tp={row.edge_tp} fp={row.edge_fp} fn={row.edge_fn} "
             f"adj={row.adj_edge_jaccard:.4f} ({time.time()-t0:.1f}s)")
    return cv_result_to_dict(aggregate(rows)), internals


def _gate(arm: dict, champ: dict) -> dict:
    champ_by = {r["name"]: r["adjusted_edge_jaccard"] for r in champ["per_dataset"]}
    arm_by = {r["name"]: r["adjusted_edge_jaccard"] for r in arm["per_dataset"]}
    no_reg = all(arm_by[n] >= champ_by[n] - 1e-9 for n in champ_by)
    delta = arm["micro_adj_edge_jaccard"] - champ["micro_adj_edge_jaccard"]
    promote = (delta > NOISE) and no_reg
    decision = "PROMOTE" if promote else (
        "REJECT" if delta < -NOISE or not no_reg else "INCONCLUSIVE")
    return {
        "micro_adj_arm": round(arm["micro_adj_edge_jaccard"], 4),
        "micro_adj_champion": round(champ["micro_adj_edge_jaccard"], 4),
        "micro_adj_delta": round(delta, 4),
        "no_regression": no_reg,
        "per_family_delta": {n: round(arm_by[n] - champ_by[n], 4) for n in champ_by},
        "decision": decision,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-thresholds", default="0.99",
                    help="comma list of detection sigmoid thresholds to sweep "
                         "(notebook default 0.99)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames (debug)")
    ap.add_argument("--out",
                    default=str(REPO / "experiments/sot3012/screen_hybrid_ab.json"))
    args = ap.parse_args(argv)

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    import predict_unet_transformer as P
    _log(f"loading released weights {WEIGHTS.name} on {device} ...")
    model_bundle = P.load_model(WEIGHTS, device)
    _log(f"model loaded (window_size={model_bundle[1]}, downsample={model_bundle[2]})")

    # Champion arm (pure classical DoG+NMS detection + champion linking).
    config = load_champion_config()
    _detect, champ_link, _scale = champion_params(config)
    _log("champion CV (classical DoG+NMS detection + champion linking)...")
    champ = cv_result_to_dict(evaluate_cv(config))
    _log(f"champion micro_adj={champ['micro_adj_edge_jaccard']}")

    thresholds = [float(s) for s in args.det_thresholds.split(",") if s.strip()]
    arms, gates, internals = {}, {}, {}
    for thr in thresholds:
        _log(f"learned-detection substrate + CHAMPION classical link, det_threshold={thr} ...")
        arm, intern = learned_detect_champion_link_cv(
            model_bundle, thr, device, args.max_frames, champ_link)
        key = f"{thr:g}"
        arms[key] = arm
        internals[key] = intern
        gates[key] = _gate(arm, champ)
        _log(f"learned-det+champ-link[thr={key}] micro_adj={arm['micro_adj_edge_jaccard']} "
             f"decision={gates[key]['decision']} no_reg={gates[key]['no_regression']}")

    best_key = max(gates, key=lambda k: gates[k]["micro_adj_delta"])
    payload = {
        "issue": "SOT-3012",
        "axis": ("role A' detection-only hybrid — official royerlab released-weights "
                 "DETECTION substrate -> CHAMPION classical ARGUS motion-model link "
                 "(linking byte据置き; learned linking NOT reintroduced)"),
        "sources": {
            "repo": "https://github.com/royerlab/kaggle-cell-tracking-competition",
            "weights_dataset": "thibautgoldsborough/cellmot-baseline-artifacts",
            "notebook": "thibautgoldsborough/unet-baseline-inference-submission",
            "infra_reused_from": "experiments/sot3011/_ext",
        },
        "leak_caveat": ("released split_0 weights trained on the labelled train "
                        "videos INCLUDING these 4 CV families -> learned-DETECTION "
                        "CV is an optimistic (train-contaminated) upper bound"),
        "champion_link_params": {
            "max_distance": champ_link.max_distance,
            "min_track_length": champ_link.min_track_length,
            "motion_model_link": champ_link.motion_model_link,
            "motion_smooth_sigma": champ_link.motion_smooth_sigma,
            "motion_gain": champ_link.motion_gain,
            "motion_gate_on_prediction": champ_link.motion_gate_on_prediction,
        },
        "noise_band": NOISE,
        "det_thresholds": thresholds,
        "champion_cv": champ,
        "learned_detect_champion_link_cv": arms,
        "detection_internals": internals,
        "gates": gates,
        "best_threshold": best_key,
        "overall_decision": (
            "PROMOTE" if any(g["decision"] == "PROMOTE" for g in gates.values())
            else "REJECT" if all(g["decision"] == "REJECT" for g in gates.values())
            else "INCONCLUSIVE"),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    for key, g in gates.items():
        _log(f"[thr={key}] decision={g['decision']} micro_delta={g['micro_adj_delta']:+.4f} "
             f"no_regression={g['no_regression']} per_family={g['per_family_delta']}")
    _log(f"OVERALL={payload['overall_decision']} best_thr={best_key}  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
