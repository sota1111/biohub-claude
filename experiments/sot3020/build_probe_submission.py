"""SOT-3020 — live-probe submission builder for the LEARNED-DETECTION candidate.

Observation probe (does NOT change the champion). Reproduces the SOT-3012
role A' hybrid **verbatim** — the official royerlab released-weights
DETECTION substrate (TemporalUNet3D, split_0) feeding the CHAMPION classical
ARGUS motion-model LAP linking — and writes a single competition submission
CSV over the four scored test videos (``data/test/*.zarr``, i.e. the exact
``CV_HOLDOUT`` families that make up the real Kaggle test set).

The point of the probe is to measure the *real* Kaggle public LB of this
learned candidate, because its internal leak-free CV is contaminated (the
released split_0 weights were trained on the labelled train videos, which
include these four families — SOT-3015: true leak-free 0.6217 vs internal
0.62->0.81). The only uncontaminated signal is the live public LB.

Config is FROZEN to the documented candidate: ``det_threshold=0.99`` (the
notebook default and SOT-3012's ``best_threshold``, internal micro_adj 0.8344).
Nothing about the strategy is changed — this is a straight strength read of the
current learned candidate. Detection centres are linked by the SAME
``link_centroids`` + champion ``LinkParams`` the champion uses (byte-identical
linking; no learned linking, no ILP), so the submission is the pure detection
substrate swapped under the champion.

Deterministic offline GPU build: torch eval + fixed released weights, no
randomness. Reuses SOT-3011's fetched external infra (clone + released weights)
under ``experiments/sot3011/_ext`` — nothing is re-downloaded.

    python experiments/sot3020/build_probe_submission.py \
        --out .targets/biohub-claude/submission.csv   # (repo-relative default)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Reuse SOT-3011's fetched external infra (clone + released weights). Identical
# detection stack to SOT-3011/3012; only the linking is the classical champion.
EXT = REPO / "experiments/sot3011/_ext/royerlab"
ART = REPO / "experiments/sot3011/_ext/artifacts/cellmot-baseline-artifacts"
WEIGHTS = ART / "weights/unet_transformer/split_0/edge_predictor_best.pth"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(EXT / "src"))
sys.path.insert(0, str(EXT / "scripts"))

import numpy as np  # noqa: E402

from biohub_tracking.champion import champion_params, load_champion_config  # noqa: E402
from biohub_tracking.eval.cv import CV_HOLDOUT  # noqa: E402
from biohub_tracking.io import geff_scale, write_submission_csv  # noqa: E402
from biohub_tracking.link import link_centroids  # noqa: E402


def _log(m: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def _detections_by_t(coords: np.ndarray) -> dict[int, np.ndarray]:
    """Group official detection ``coords[N,4]=(t,z,y,x)`` into the
    ``{t: (M,3) zyx}`` mapping ``link_centroids`` expects (row order preserved,
    matching the deterministic id assignment ``link_centroids`` performs)."""
    by_t: dict[int, list] = defaultdict(list)
    for t, z, y, x in coords:
        by_t[int(t)].append((float(z), float(y), float(x)))
    return {t: np.asarray(v, dtype=float) for t, v in sorted(by_t.items())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--det-threshold", type=float, default=0.99,
                    help="detection sigmoid threshold (frozen candidate = notebook "
                         "default 0.99 = SOT-3012 best_threshold, internal 0.8344)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames (debug only)")
    ap.add_argument("--out", default=str(REPO / "submission.csv"),
                    help="output submission CSV path")
    ap.add_argument("--meta-out", default=str(REPO / "experiments/sot3020/build_meta.json"),
                    help="fingerprint / provenance sidecar")
    args = ap.parse_args(argv)

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    import predict_unet_transformer as P
    _log(f"loading released weights {WEIGHTS.name} on {device} ...")
    model, window_size, downsample = P.load_model(WEIGHTS, device)
    _log(f"model loaded (window_size={window_size}, downsample={downsample})")

    # CHAMPION classical linking params, byte据置き (motion-model LAP,
    # motion_gain=1.0, min_track_length=4). Detection is the ONLY swapped stage.
    config = load_champion_config()
    _detect, champ_link, _scale = champion_params(config)

    cfg = P.PredictConfig(det_threshold=args.det_threshold, use_ilp=False)
    graphs = {}
    internals = []
    for fam in CV_HOLDOUT:
        image = REPO / fam.image
        # Physical voxel scale of this video (same physical volume as the train
        # geff; identical to the SOT-3012 A/B arm which used geff_scale).
        scale = geff_scale(REPO / fam.geff)
        t0 = time.time()
        coords, _learned_edges = P.predict_video(
            model, image, device, cfg=cfg, window_size=window_size,
            max_frames=args.max_frames, downsample=downsample,
        )
        n_det = len(coords)
        detections = _detections_by_t(coords)
        pred = link_centroids(detections, scale=scale, params=champ_link)
        graphs[fam.name] = pred
        internals.append({
            "name": fam.name,
            "n_det_learned": int(n_det),
            "pred_nodes": int(pred.num_nodes),
            "pred_edges": int(pred.num_edges),
        })
        _log(f"  [{fam.name}] det={n_det} kept_nodes={pred.num_nodes} "
             f"edges={pred.num_edges} ({time.time()-t0:.1f}s)")

    out = write_submission_csv(graphs, args.out)
    total_nodes = sum(g.num_nodes for g in graphs.values())
    total_edges = sum(g.num_edges for g in graphs.values())
    _log(f"wrote {out}: {len(graphs)} datasets, {total_nodes} nodes, {total_edges} edges")

    meta = {
        "issue": "SOT-3020",
        "candidate": "SOT-3012 role A' hybrid — learned DETECTION substrate "
                     "(TemporalUNet3D released split_0) -> CHAMPION classical "
                     "ARGUS motion-model link (ILP-free, PORTABLE)",
        "det_threshold": args.det_threshold,
        "champion_link_params": {
            "max_distance": champ_link.max_distance,
            "min_track_length": champ_link.min_track_length,
            "motion_model_link": champ_link.motion_model_link,
            "motion_smooth_sigma": champ_link.motion_smooth_sigma,
            "motion_gain": champ_link.motion_gain,
            "motion_gate_on_prediction": champ_link.motion_gate_on_prediction,
        },
        "weights": str(WEIGHTS.relative_to(REPO)),
        "device": str(device),
        "internal_micro_adj_at_0.99": 0.8344,
        "leak_caveat": "released split_0 weights trained on the labelled train "
                       "videos INCLUDING these 4 test families -> internal CV is a "
                       "train-contaminated optimistic upper bound (SOT-3015).",
        "datasets": internals,
        "total_nodes": int(total_nodes),
        "total_edges": int(total_edges),
    }
    Path(args.meta_out).write_text(json.dumps(meta, indent=2) + "\n")
    _log(f"wrote meta {args.meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
