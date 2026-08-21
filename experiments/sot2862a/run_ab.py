"""SOT-2863 (cycle-9 child A) — leak-free CV A/B: PU-aware learned detectors vs
naive-loss (SOT-2848) vs classical champion.

Detection-only swap, linking fixed (champion distance-only bipartite): every arm
runs the SAME leak-free harness (``biohub_tracking.eval.cv``), differing ONLY in
the per-voxel detector. Arms scored:

* ``champion``   — classical DoG+NMS via the harness ``evaluate_cv()``.
* ``naive``      — SOT-2848's foreground-weighted-MSE weights
  (``experiments/sot2848/weights/<family>.pt``) — the same-seed baseline learned arm.
* ``nnpu``       — SOT-2863 nnPU-corrected-loss weights.
* ``cellsparse`` — SOT-2863 Cellsparse ignore-weighting weights.

The learned arms share the champion linker and the same LOFO fold assignment, so
the A/B isolates the training loss. Promotion gate (CLAUDE.md / SOT-2817): a
learned arm is promoted only if it beats the champion micro-adj beyond the noise
band AND regresses no single family. Otherwise the champion stays byte-frozen and
the arm ships default-off. Runs in the repo ``.venv`` (torch + scipy + zarr).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from biohub_tracking.champion import champion_params, load_champion_config  # noqa: E402
from biohub_tracking.eval.cv import (  # noqa: E402
    CV_HOLDOUT,
    CHAMPION_REFERENCE_MICRO_ADJ,
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
from biohub_tracking.learned_detect import LearnedDetector, LearnedDetectorConfig  # noqa: E402
from biohub_tracking.link import link_centroids  # noqa: E402
from biohub_tracking.pipeline import _open_image_array  # noqa: E402

NOISE = 0.003

ARM_WEIGHT_DIRS = {
    "naive": REPO / "experiments/sot2848/weights",
    "nnpu": REPO / "experiments/sot2862a/weights/nnpu",
    "cellsparse": REPO / "experiments/sot2862a/weights/cellsparse",
}


def _build_detector(weights: Path, threshold: float, device: str) -> LearnedDetector:
    cfg = LearnedDetectorConfig(
        enabled=True, arch="temporal_unet3d", weights=None,
        threshold=threshold, nms_size_zyx=(2, 5, 5), device=device,
    )
    return LearnedDetector(cfg).load(weights_path=weights)


def learned_cv(weights_dir: Path, threshold: float, device: str, max_t: int | None, log,
               label: str) -> dict:
    _detect, link, _scale = champion_params(load_champion_config())
    rows = []
    for fam in CV_HOLDOUT:
        weights = weights_dir / f"{fam.name}.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"missing LOFO weights for {fam.name}: {weights}")
        det = _build_detector(weights, threshold, device)
        arr = _open_image_array(REPO / fam.image)
        t0 = time.time()
        detections = det.detect_series(arr, max_t=max_t)
        pred = link_centroids(detections, scale=geff_scale(REPO / fam.geff), params=link)
        gt = load_geff(REPO / fam.geff)
        n_true = geff_estimated_num_nodes(REPO / fam.geff)
        row = score_family(fam, pred, gt, n_true, scale=geff_scale(REPO / fam.geff))
        n_det = sum(len(v) for v in detections.values())
        log(f"  [{label}|{fam.name}] thr={threshold} det={n_det} pred_nodes={row.num_pred_nodes} "
            f"tp={row.edge_tp} fp={row.edge_fp} fn={row.edge_fn} adj={row.adj_edge_jaccard:.4f} "
            f"({time.time()-t0:.1f}s)")
        rows.append(row)
    return cv_result_to_dict(aggregate(rows))


def _gate(arm: dict, champ: dict) -> dict:
    """Promotion decision for a learned arm vs the champion (micro-adj + no-reg)."""
    champ_by_family = {r["name"]: r["adjusted_edge_jaccard"] for r in champ["per_dataset"]}
    arm_by_family = {r["name"]: r["adjusted_edge_jaccard"] for r in arm["per_dataset"]}
    no_regression = all(arm_by_family[n] >= champ_by_family[n] - 1e-9 for n in champ_by_family)
    micro_delta = arm["micro_adj_edge_jaccard"] - champ["micro_adj_edge_jaccard"]
    promote = (micro_delta > NOISE) and no_regression
    decision = "PROMOTE" if promote else (
        "REJECT" if micro_delta < -NOISE or not no_regression else "INCONCLUSIVE")
    return {
        "micro_adj_arm": arm["micro_adj_edge_jaccard"],
        "micro_adj_champion": champ["micro_adj_edge_jaccard"],
        "micro_adj_delta": round(micro_delta, 4),
        "no_regression": no_regression,
        "per_family_delta": {n: round(arm_by_family[n] - champ_by_family[n], 4)
                             for n in champ_by_family},
        "decision": decision,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="naive,nnpu,cellsparse",
                    help="comma list of learned arms to score")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="committed operating point for the promotion decision")
    ap.add_argument("--sweep", type=str, default="0.3,0.5,0.7",
                    help="diagnostic threshold sweep (NOT used for promotion)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-t", type=int, default=None, help="cap timepoints (debug)")
    ap.add_argument("--out", default=str(REPO / "experiments/sot2862a/screen_pu_ab.json"))
    ap.add_argument("--tag", default="screen")
    args = ap.parse_args(argv)

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    def log(m):
        print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    log("champion CV (classical DoG+NMS, harness evaluate_cv)...")
    champ = cv_result_to_dict(evaluate_cv())
    log(f"champion micro_adj={champ['micro_adj_edge_jaccard']} (ref {CHAMPION_REFERENCE_MICRO_ADJ})")

    committed = {}
    sweeps = {}
    gates = {}
    for arm in arms:
        wd = ARM_WEIGHT_DIRS[arm]
        log(f"{arm} CV at committed threshold={args.threshold} (weights {wd})...")
        committed[arm] = learned_cv(wd, args.threshold, args.device, args.max_t, log, arm)
        log(f"{arm} micro_adj={committed[arm]['micro_adj_edge_jaccard']}")
        gates[arm] = _gate(committed[arm], champ)
        arm_sweep = {}
        for thr in [float(s) for s in args.sweep.split(",") if s.strip()]:
            if thr == args.threshold:
                arm_sweep[str(thr)] = committed[arm]
                continue
            log(f"{arm} diagnostic sweep threshold={thr}...")
            arm_sweep[str(thr)] = learned_cv(wd, thr, args.device, args.max_t, log, arm)
        sweeps[arm] = arm_sweep

    # nnPU-vs-naive same-seed attribution (the failure-attribution the issue demands).
    nnpu_vs_naive = None
    if "nnpu" in committed and "naive" in committed:
        nnpu_vs_naive = round(
            committed["nnpu"]["micro_adj_edge_jaccard"]
            - committed["naive"]["micro_adj_edge_jaccard"], 4)

    payload = {
        "tag": args.tag,
        "committed_threshold": args.threshold,
        "noise_band": NOISE,
        "champion_cv": champ,
        "committed_cv": committed,
        "diagnostic_sweep": sweeps,
        "gates": gates,
        "nnpu_vs_naive_micro_delta": nnpu_vs_naive,
        "overall_decision": (
            "PROMOTE" if any(g["decision"] == "PROMOTE" for g in gates.values())
            else "REJECT" if any(g["decision"] == "REJECT" for g in gates.values())
            else "INCONCLUSIVE"),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    for arm, g in gates.items():
        log(f"[{arm}] decision={g['decision']} micro_delta={g['micro_adj_delta']:+.4f} "
            f"no_regression={g['no_regression']}")
    if nnpu_vs_naive is not None:
        log(f"nnpu_vs_naive micro_delta={nnpu_vs_naive:+.4f}")
    log(f"OVERALL={payload['overall_decision']}")
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
