"""SOT-2272 screen: sweep detection threshold_percentile on BOTH local GT families.

For each threshold, build the champion-style submission over both local test
videos and score the micro-averaged adjusted edge Jaccard against both GT geffs.
The champion (99.3) is the incumbent; we look for a threshold that lifts the
catastrophic fragmented family (44b6_0b24845f) without wrecking the clean one.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
from biohub_tracking.detect import DetectParams
from biohub_tracking.link import LinkParams
from biohub_tracking.pipeline import run_pipeline
from biohub_tracking.io import load_geff, geff_scale, geff_estimated_num_nodes
from biohub_tracking.eval.score import evaluate, adjusted_edge_jaccard, _jaccard

FAMILIES = ["44b6_0113de3b", "44b6_0b24845f"]
SCALE = (1.625, 0.40625, 0.40625)

def score_at(pct, max_distance=7.0, allow_division=False, sigma=(1.0,3.0,3.0), nms=(2,5,5)):
    dp = DetectParams(sigma_zyx=sigma, nms_size_zyx=nms, threshold_percentile=pct)
    lp = LinkParams(max_distance=max_distance, allow_division=allow_division)
    per = {}
    wsum = 0.0; wtot = 0.0
    for name in FAMILIES:
        g = run_pipeline(f"data/test/{name}.zarr", scale=SCALE, detect_params=dp, link_params=lp)
        gt = load_geff(f"data/train/{name}.geff")
        n_true = geff_estimated_num_nodes(f"data/train/{name}.geff")
        r = evaluate(g, gt, scale=SCALE, max_distance=max_distance)
        j = _jaccard(r.edge_tp, r.edge_fp, r.edge_fn)
        adj = adjusted_edge_jaccard(j, r.num_pred_nodes, n_true)
        if adj != adj: adj = j
        w = r.edge_tp + r.edge_fp + r.edge_fn
        if w > 0: wsum += w*adj; wtot += w
        per[name] = dict(tp=r.edge_tp, fp=r.edge_fp, fn=r.edge_fn, jacc=round(j,4),
                         adj=round(adj,4), pred_nodes=r.num_pred_nodes, n_true=n_true)
    micro = wsum/wtot if wtot>0 else float("nan")
    return micro, per

if __name__ == "__main__":
    grid = [float(x) for x in sys.argv[1:]] or [99.3, 95.0, 90.0, 85.0, 80.0, 75.0, 70.0]
    out = []
    for pct in grid:
        t0=time.time()
        micro, per = score_at(pct)
        dt=time.time()-t0
        row = dict(pct=pct, micro=round(micro,4), per=per, secs=round(dt,1))
        out.append(row)
        print(f"pct={pct:5.1f} micro_adj={micro:.4f} "
              f"clean(tp/fp/fn={per['44b6_0113de3b']['tp']}/{per['44b6_0113de3b']['fp']}/{per['44b6_0113de3b']['fn']} adj={per['44b6_0113de3b']['adj']}) "
              f"frag(tp/fp/fn={per['44b6_0b24845f']['tp']}/{per['44b6_0b24845f']['fp']}/{per['44b6_0b24845f']['fn']} adj={per['44b6_0b24845f']['adj']} nodes={per['44b6_0b24845f']['pred_nodes']}) "
              f"[{dt:.0f}s]", flush=True)
        Path("experiments/sot2272/screen_threshold.json").write_text(json.dumps(out, indent=2))
    print("DONE")
