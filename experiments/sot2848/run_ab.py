"""SOT-2848 — leak-free CV A/B: learned TemporalUNet3D detector vs classical champion.

Detection-only swap, linking fixed (champion distance-only bipartite): for each
holdout family we run the ``temporal_unet3d`` detector whose weights were trained
**without that family** (``train_lofo.py`` → ``weights/<family>.pt``), link with the
champion linker, and score through the single leak-free CV harness
(``biohub_tracking.eval.cv``). The champion arm is the harness's own
``evaluate_cv()``. Both arms are scored by identical code, so the only difference
is classical DoG+NMS detection vs the learned heatmap detector.

Promotion gate (CLAUDE.md / SOT-2817): the learned arm is promoted only if it
beats the champion micro-adj by more than noise AND regresses no single family
(``CvResult.no_regression_vs``). Otherwise the champion stays byte-frozen.

Runs in the repo ``.venv`` (torch + scipy + zarr).
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


def _build_detector(weights: Path, threshold: float, device: str) -> LearnedDetector:
    cfg = LearnedDetectorConfig(
        enabled=True, arch="temporal_unet3d", weights=None,
        threshold=threshold, nms_size_zyx=(2, 5, 5), device=device,
    )
    return LearnedDetector(cfg).load(weights_path=weights)


def learned_cv(threshold: float, device: str, max_t: int | None, log) -> dict:
    _detect, link, _scale = champion_params(load_champion_config())
    rows = []
    for fam in CV_HOLDOUT:
        weights = REPO / f"experiments/sot2848/weights/{fam.name}.pt"
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
        log(f"  [{fam.name}] thr={threshold} detections={n_det} pred_nodes={row.num_pred_nodes} "
            f"edge_tp={row.edge_tp} fp={row.edge_fp} fn={row.edge_fn} adj={row.adj_edge_jaccard:.4f} "
            f"({time.time()-t0:.1f}s)")
        rows.append(row)
    return cv_result_to_dict(aggregate(rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="committed operating point for the promotion decision")
    ap.add_argument("--sweep", type=str, default="0.3,0.5,0.7",
                    help="diagnostic threshold sweep (NOT used for promotion)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-t", type=int, default=None, help="cap timepoints (debug)")
    ap.add_argument("--out", default=str(REPO / "experiments/sot2848/screen_learned_ab.json"))
    ap.add_argument("--tag", default="screen")
    args = ap.parse_args(argv)

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    def log(m):
        print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)

    log("champion CV (classical DoG+NMS, harness evaluate_cv)...")
    champ = cv_result_to_dict(evaluate_cv())
    log(f"champion micro_adj={champ['micro_adj_edge_jaccard']} (ref {CHAMPION_REFERENCE_MICRO_ADJ})")

    log(f"learned CV at committed threshold={args.threshold}...")
    learned = learned_cv(args.threshold, args.device, args.max_t, log)
    log(f"learned micro_adj={learned['micro_adj_edge_jaccard']}")

    sweep = {}
    for thr in [float(s) for s in args.sweep.split(",") if s.strip()]:
        if thr == args.threshold:
            sweep[str(thr)] = learned
            continue
        log(f"diagnostic sweep threshold={thr}...")
        sweep[str(thr)] = learned_cv(thr, args.device, args.max_t, log)

    # Promotion gate: micro-adj beats champion beyond noise AND no per-family regression.
    champ_by_family = {r["name"]: r["adjusted_edge_jaccard"] for r in champ["per_dataset"]}
    learned_by_family = {r["name"]: r["adjusted_edge_jaccard"] for r in learned["per_dataset"]}
    no_regression = all(
        learned_by_family[n] >= champ_by_family[n] - 1e-9 for n in champ_by_family
    )
    micro_delta = learned["micro_adj_edge_jaccard"] - champ["micro_adj_edge_jaccard"]
    NOISE = 0.003
    promote = (micro_delta > NOISE) and no_regression
    decision = "PROMOTE" if promote else ("REJECT" if micro_delta < -NOISE or not no_regression else "INCONCLUSIVE")

    payload = {
        "tag": args.tag,
        "committed_threshold": args.threshold,
        "champion_cv": champ,
        "learned_cv": learned,
        "diagnostic_sweep": sweep,
        "ab": {
            "micro_adj_champion": champ["micro_adj_edge_jaccard"],
            "micro_adj_learned": learned["micro_adj_edge_jaccard"],
            "micro_adj_delta": round(micro_delta, 4),
            "no_regression": no_regression,
            "per_family_delta": {
                n: round(learned_by_family[n] - champ_by_family[n], 4) for n in champ_by_family
            },
            "noise_band": NOISE,
            "decision": decision,
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    log(f"DECISION={decision} micro_delta={micro_delta:+.4f} no_regression={no_regression}")
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
