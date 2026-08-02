"""SOT-2272 alt-detector screen: TOP-K peaks per frame (adaptive to true count).

Instead of a global percentile threshold (which can't see a dim cell at ~p60
without flooding bright videos), keep the K brightest NMS peaks per frame, with
K scaled to the per-video estimated true node count (N_true / n_frames). This
adapts detection density to each video and pushes N_pred toward N_true (which the
adjusted-jaccard node-count term rewards). Scored on both local GT families.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, "src")
from biohub_tracking.link import LinkParams, link_centroids
from biohub_tracking.io import load_geff, geff_scale, geff_estimated_num_nodes
from biohub_tracking.eval.score import evaluate, adjusted_edge_jaccard, _jaccard
from biohub_tracking.pipeline import _open_image_array

FAMILIES = ["44b6_0113de3b", "44b6_0b24845f"]
SCALE = (1.625, 0.40625, 0.40625)
SIGMA = (1.0, 3.0, 3.0)
NMS = (2, 5, 5)

def detect_topk(vol, k, min_frac=0.0):
    vol = np.asarray(vol, dtype=np.float32)
    sm = ndi.gaussian_filter(vol, sigma=SIGMA)
    footprint = np.ones([2*s+1 for s in NMS], dtype=bool)
    local_max = ndi.maximum_filter(sm, footprint=footprint)
    peak_mask = (sm == local_max)
    if min_frac > 0:
        peak_mask &= (sm > np.percentile(sm, min_frac))
    coords = np.argwhere(peak_mask).astype(np.float64)
    if coords.size == 0:
        return coords.reshape(0, 3)
    inten = sm[peak_mask]
    order = np.argsort(inten)[::-1]
    coords = coords[order]
    if k is not None and len(coords) > k:
        coords = coords[:k]
    return coords

def score_topk(k_per_frame_factor=1.0, max_distance=7.0, min_frac=0.0):
    """k per frame = factor * (N_true / n_frames_video)."""
    wsum=0.0; wtot=0.0; per={}
    for name in FAMILIES:
        arr = _open_image_array(f"data/test/{name}.zarr")
        n_frames = arr.shape[0]
        n_true = geff_estimated_num_nodes(f"data/train/{name}.geff")
        k = int(round(k_per_frame_factor * n_true / n_frames))
        dets = {t: detect_topk(arr[t], k, min_frac) for t in range(n_frames)}
        g = link_centroids(dets, scale=SCALE, params=LinkParams(max_distance=max_distance, allow_division=False))
        gt = load_geff(f"data/train/{name}.geff")
        r = evaluate(g, gt, scale=SCALE, max_distance=max_distance)
        j = _jaccard(r.edge_tp, r.edge_fp, r.edge_fn)
        adj = adjusted_edge_jaccard(j, r.num_pred_nodes, n_true)
        if adj != adj: adj = j
        w = r.edge_tp+r.edge_fp+r.edge_fn
        if w>0: wsum+=w*adj; wtot+=w
        per[name]=dict(tp=r.edge_tp,fp=r.edge_fp,fn=r.edge_fn,adj=round(adj,4),
                       pred_nodes=r.num_pred_nodes,n_true=n_true,k=k)
    return (wsum/wtot if wtot>0 else float("nan")), per

if __name__ == "__main__":
    grid = [float(x) for x in sys.argv[1:]] or [0.5, 1.0, 1.5, 2.0]
    out=[]
    for f in grid:
        t0=time.time(); micro,per=score_topk(f); dt=time.time()-t0
        out.append(dict(factor=f,micro=round(micro,4),per=per,secs=round(dt,1)))
        print(f"factor={f:4.1f} micro_adj={micro:.4f} "
              f"clean(tp/fp/fn={per['44b6_0113de3b']['tp']}/{per['44b6_0113de3b']['fp']}/{per['44b6_0113de3b']['fn']} adj={per['44b6_0113de3b']['adj']} k={per['44b6_0113de3b']['k']}) "
              f"frag(tp/fp/fn={per['44b6_0b24845f']['tp']}/{per['44b6_0b24845f']['fp']}/{per['44b6_0b24845f']['fn']} adj={per['44b6_0b24845f']['adj']} k={per['44b6_0b24845f']['k']} nodes={per['44b6_0b24845f']['pred_nodes']}) "
              f"[{dt:.0f}s]", flush=True)
        Path("experiments/sot2272/screen_topk.json").write_text(json.dumps(out,indent=2))
    print("DONE")
