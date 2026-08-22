"""SOT-3011 — leak-free CV A/B: the OFFICIAL royerlab learned pipeline adopted
WHOLESALE via its RELEASED pretrained weights (TemporalUNet3D detection +
SimpleNodeTransformer edge prediction), run offline, vs the classical DoG+NMS
champion (micro_adj 0.6760).

Role A' *wholesale* (SOT-3010 cycle-8 direction 1). The new evidence over the
three prior learned-detector/linker REJECTs (SOT-2828/2848/2863 detector,
SOT-2841/2870/2994 linking, SOT-2993 from-scratch masked UNet 0.6217) is that
this uses the ORGANISER's CONVERGED public weights — not a from-scratch / under-
converged proxy — so it measures learned detection+linking at its real operating
point.

Both arms are scored by the SAME leak-free harness (``biohub_tracking.eval.cv``),
so the A/B isolates the pipeline. The learned arm reuses the official inference
code verbatim (``predict_unet_transformer.predict_video`` + ``load_model``) from a
git clone of ``royerlab/kaggle-cell-tracking-competition`` and the released
weights dataset ``thibautgoldsborough/cellmot-baseline-artifacts``; its per-video
graph is converted to our ``TrackingGraph`` and scored with the identical
``score_family``/``aggregate`` the champion uses.

LEAK CAVEAT (recorded honestly): the released split_0 weights were trained on the
organiser's labelled train videos, which INCLUDE these four CV families. So the
learned arm's CV here is an *optimistic upper bound* (train-contaminated), not a
leak-free estimate. That only strengthens a REJECT and heavily caveats any PASS.

No Kaggle submission (child issue; submission is the parent's call).
Runs in the repo ``.venv`` (torch+cuda + zarr + tracksdata + pyscipopt).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "experiments/sot3011/_ext/royerlab"
ART = REPO / "experiments/sot3011/_ext/artifacts/cellmot-baseline-artifacts"
WEIGHTS = ART / "weights/unet_transformer/split_0/edge_predictor_best.pth"

sys.path.insert(0, str(REPO / "src"))
# Official inference code (clone) + its sibling scripts.
sys.path.insert(0, str(EXT / "src"))
sys.path.insert(0, str(EXT / "scripts"))

from biohub_tracking.eval.cv import (  # noqa: E402
    CV_HOLDOUT,
    CHAMPION_REFERENCE_MICRO_ADJ,
    aggregate,
    cv_result_to_dict,
    evaluate_cv,
    score_family,
)
from biohub_tracking.graph import TrackingGraph  # noqa: E402
from biohub_tracking.io import (  # noqa: E402
    geff_estimated_num_nodes,
    geff_scale,
    load_geff,
)

NOISE = 0.003


def _log(m: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def _pred_from_coords_edges(coords, edges) -> TrackingGraph:
    """Build our TrackingGraph from official (coords[N,4]=t,z,y,x, edges list).

    Edge tuples are ``(src_global_idx, tgt_global_idx, prob, dist)`` where the
    indices are rows of ``coords`` (original-resolution voxel space — the same
    grid the GT geff nodes live in, so the metric's scale handling matches)."""
    g = TrackingGraph()
    for i, (t, z, y, x) in enumerate(coords):
        g.add_node(i, float(t), float(z), float(y), float(x))
    for e in edges:
        src, tgt = int(e[0]), int(e[1])
        g.add_edge(src, tgt)
    return g


def _pred_from_td_graph(graph) -> TrackingGraph:
    """Convert a solved tracksdata InMemoryGraph to our TrackingGraph."""
    g = TrackingGraph()
    nrows = list(graph.node_attrs().iter_rows(named=True))
    for r in nrows:
        g.add_node(int(r["node_id"]), float(r["t"]), float(r["z"]),
                   float(r["y"]), float(r["x"]))
    have = set(g.coords)
    for r in graph.edge_attrs().iter_rows(named=True):
        s, t = int(r["source_id"]), int(r["target_id"])
        if s in have and t in have:
            g.add_edge(s, t)
    return g


def learned_cv(model_bundle, cfg, linker: str, device, max_frames, ilp_cfg) -> dict:
    import predict_unet_transformer as P  # noqa: E402

    model, window_size, downsample = model_bundle
    rows = []
    for fam in CV_HOLDOUT:
        ds_path = REPO / fam.image
        t0 = time.time()
        coords, edges = P.predict_video(
            model, ds_path, device, cfg=cfg, window_size=window_size,
            max_frames=max_frames, downsample=downsample,
        )
        n_det = len(coords)
        if linker == "ilp" and len(edges) > 0:
            import tracksdata as td
            graph = P.build_graph(coords, edges)
            solver = td.solvers.ILPSolver(
                edge_weight=ilp_cfg["edge_weight"] * td.EdgeAttr("edge_prob"),
                appearance_weight=ilp_cfg["appearance_weight"],
                disappearance_weight=ilp_cfg["disappearance_weight"],
                division_weight=ilp_cfg["division_weight"],
            )
            solved = solver.solve(graph)
            pred = _pred_from_td_graph(solved)
        else:
            pred = _pred_from_coords_edges(coords, edges)

        gt = load_geff(REPO / fam.geff)
        n_true = geff_estimated_num_nodes(REPO / fam.geff)
        row = score_family(fam, pred, gt, n_true, scale=geff_scale(REPO / fam.geff))
        rows.append(row)
        _log(f"  [learned:{linker}|{fam.name}] det={n_det} pred_edges={pred.num_edges} "
             f"tp={row.edge_tp} fp={row.edge_fp} fn={row.edge_fn} "
             f"adj={row.adj_edge_jaccard:.4f} ({time.time()-t0:.1f}s)")
    return cv_result_to_dict(aggregate(rows))


def _gate(arm: dict, champ: dict) -> dict:
    champ_by = {r["name"]: r["adjusted_edge_jaccard"] for r in champ["per_dataset"]}
    arm_by = {r["name"]: r["adjusted_edge_jaccard"] for r in arm["per_dataset"]}
    no_reg = all(arm_by[n] >= champ_by[n] - 1e-9 for n in champ_by)
    delta = arm["micro_adj_edge_jaccard"] - champ["micro_adj_edge_jaccard"]
    promote = (delta > NOISE) and no_reg
    decision = "PROMOTE" if promote else (
        "REJECT" if delta < -NOISE or not no_reg else "INCONCLUSIVE")
    return {
        "micro_adj_arm": arm["micro_adj_edge_jaccard"],
        "micro_adj_champion": champ["micro_adj_edge_jaccard"],
        "micro_adj_delta": round(delta, 4),
        "no_regression": no_reg,
        "per_family_delta": {n: round(arm_by[n] - champ_by[n], 4) for n in champ_by},
        "decision": decision,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--linkers", default="greedy",
                    help="comma list of learned linkers to score: greedy,ilp")
    ap.add_argument("--det-threshold", type=float, default=0.99,
                    help="detection peak sigmoid threshold (notebook default 0.99)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames (debug)")
    ap.add_argument("--out", default=str(REPO / "experiments/sot3011/screen_royerlab_ab.json"))
    args = ap.parse_args(argv)

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    import predict_unet_transformer as P
    _log(f"loading released weights {WEIGHTS.name} on {device} ...")
    model_bundle = P.load_model(WEIGHTS, device)
    _log(f"model loaded (window_size={model_bundle[1]}, downsample={model_bundle[2]})")

    ilp_cfg = {"edge_weight": -1.0, "appearance_weight": 0.1,
               "disappearance_weight": 0.1, "division_weight": 1.0}

    _log("champion CV (classical DoG+NMS, harness evaluate_cv)...")
    champ = cv_result_to_dict(evaluate_cv())
    _log(f"champion micro_adj={champ['micro_adj_edge_jaccard']} (ref {CHAMPION_REFERENCE_MICRO_ADJ})")

    linkers = [s.strip() for s in args.linkers.split(",") if s.strip()]
    arms, gates = {}, {}
    for linker in linkers:
        cfg = P.PredictConfig(det_threshold=args.det_threshold,
                              use_ilp=(linker == "ilp"))
        _log(f"learned arm linker={linker} det_threshold={args.det_threshold} ...")
        arms[linker] = learned_cv(model_bundle, cfg, linker, device,
                                  args.max_frames, ilp_cfg)
        gates[linker] = _gate(arms[linker], champ)
        _log(f"learned:{linker} micro_adj={arms[linker]['micro_adj_edge_jaccard']} "
             f"decision={gates[linker]['decision']}")

    payload = {
        "issue": "SOT-3011",
        "axis": "role A' wholesale — official royerlab released-weights pipeline",
        "sources": {
            "repo": "https://github.com/royerlab/kaggle-cell-tracking-competition",
            "weights_dataset": "thibautgoldsborough/cellmot-baseline-artifacts",
            "notebook": "thibautgoldsborough/unet-baseline-inference-submission",
        },
        "leak_caveat": ("released split_0 weights trained on the labelled train "
                        "videos INCLUDING these 4 CV families -> learned CV is an "
                        "optimistic (train-contaminated) upper bound"),
        "det_threshold": args.det_threshold,
        "noise_band": NOISE,
        "effective_config": {
            "unet_out_channels": 32, "unet_layers": [32, 64, 128],
            "downsample": [1, 4, 4], "window_size": 2,
            "det_tta": True, "edge_activation": "softmax", "edge_threshold": 0.5,
            "ilp": ilp_cfg,
        },
        "champion_cv": champ,
        "learned_cv": arms,
        "gates": gates,
        "overall_decision": (
            "PROMOTE" if any(g["decision"] == "PROMOTE" for g in gates.values())
            else "REJECT" if all(g["decision"] == "REJECT" for g in gates.values())
            else "INCONCLUSIVE"),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    for linker, g in gates.items():
        _log(f"[{linker}] decision={g['decision']} micro_delta={g['micro_adj_delta']:+.4f} "
             f"no_regression={g['no_regression']}")
    _log(f"OVERALL={payload['overall_decision']}  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
