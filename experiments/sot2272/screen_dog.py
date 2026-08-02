"""SOT-2272 DoG detector screen: background-normalized blob detection.

The champion thresholds raw smoothed intensity, so a globally dim cell (the
tracked cell in 44b6_0b24845f sits at ~p60 of the smoothed volume) is never a
top peak. A Difference-of-Gaussians response detects compact blobs brighter than
their *local* surround regardless of absolute brightness. We NMS-peak the DoG
response and keep peaks above a percentile of the (positive) DoG response.

Screened on both local GT families: detection recall on the dim cell AND the full
edge score, vs the champion baseline (clean adj 0.9512 / frag adj 0.0436 /
micro 0.5063).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, "src")
from biohub_tracking.link import LinkParams, link_centroids
from biohub_tracking.io import load_geff, geff_estimated_num_nodes
from biohub_tracking.eval.score import evaluate, adjusted_edge_jaccard, _jaccard
from biohub_tracking.pipeline import _open_image_array

FAMILIES = ["44b6_0113de3b", "44b6_0b24845f"]
SCALE = (1.625, 0.40625, 0.40625)

def detect_dog(vol, sigma_small, sigma_large, nms, pct, min_dog=0.0):
    vol = np.asarray(vol, dtype=np.float32)
    s1 = ndi.gaussian_filter(vol, sigma=sigma_small)
    s2 = ndi.gaussian_filter(vol, sigma=sigma_large)
    dog = s1 - s2
    footprint = np.ones([2*s+1 for s in nms], dtype=bool)
    lm = ndi.maximum_filter(dog, footprint=footprint)
    thr = max(float(np.percentile(dog, pct)), min_dog)
    mask = (dog == lm) & (dog > thr)
    coords = np.argwhere(mask).astype(np.float64)
    if coords.size == 0:
        return coords.reshape(0,3)
    order = np.argsort(dog[mask])[::-1]
    return coords[order]

def recall_only(sigma_small, sigma_large, nms, pct):
    """Fast: detection recall of the dim GT cell on 44b6_0b24845f."""
    name="44b6_0b24845f"
    gt=load_geff(f"data/train/{name}.geff"); arr=_open_image_array(f"data/test/{name}.zarr")
    gt_bt=gt.nodes_by_time(); scale=np.array(SCALE)
    tot=0;matched=0;ndet=0;nf=0
    for t in sorted(gt_bt):
        det=detect_dog(arr[t],sigma_small,sigma_large,nms,pct); ndet+=len(det); nf+=1
        gp=np.array([gt.position(i) for i in gt_bt[t]])*scale
        if len(det): d=np.linalg.norm(gp[:,None,:]-det[None,:,:]*scale,axis=2).min(axis=1)
        else: d=np.full(len(gt_bt[t]),1e9)
        tot+=len(gt_bt[t]); matched+=(d<=7.0).sum()
    return matched,tot,ndet/nf

def full_score(sigma_small, sigma_large, nms, pct, max_distance=7.0):
    wsum=0.0;wtot=0.0;per={}
    for name in FAMILIES:
        arr=_open_image_array(f"data/test/{name}.zarr"); nfr=arr.shape[0]
        dets={t:detect_dog(arr[t],sigma_small,sigma_large,nms,pct) for t in range(nfr)}
        g=link_centroids(dets,scale=SCALE,params=LinkParams(max_distance=max_distance,allow_division=False))
        gt=load_geff(f"data/train/{name}.geff"); n_true=geff_estimated_num_nodes(f"data/train/{name}.geff")
        r=evaluate(g,gt,scale=SCALE,max_distance=max_distance)
        j=_jaccard(r.edge_tp,r.edge_fp,r.edge_fn); adj=adjusted_edge_jaccard(j,r.num_pred_nodes,n_true)
        if adj!=adj: adj=j
        w=r.edge_tp+r.edge_fp+r.edge_fn
        if w>0: wsum+=w*adj; wtot+=w
        per[name]=dict(tp=r.edge_tp,fp=r.edge_fp,fn=r.edge_fn,adj=round(adj,4),pred_nodes=r.num_pred_nodes,n_true=n_true)
    return (wsum/wtot if wtot>0 else float("nan")),per

if __name__=="__main__":
    mode = sys.argv[1] if len(sys.argv)>1 else "recall"
    nms=(2,5,5)
    if mode=="recall":
        # sweep sigma pairs and pct, recall-only (fast) on the dim family
        configs=[
            ((1,2,2),(2,6,6)),((1,2,2),(3,8,8)),((1,3,3),(2,6,6)),
            ((1,3,3),(3,9,9)),((1,1.5,1.5),(2,5,5)),((0.8,1.5,1.5),(2,6,6)),
        ]
        for ss,sl in configs:
            for pct in [99.0,98.0,95.0,90.0]:
                m,tot,avg=recall_only(ss,sl,nms,pct)
                print(f"ss={ss} sl={sl} pct={pct:4.1f} recall={m:2d}/{tot} ({m/tot:.2f}) avgdet/f={avg:.0f}",flush=True)
        print("DONE")
    else:
        # full score for promising configs (args: ss1,ss2,ss3 sl1,sl2,sl3 pct)
        grid=json.loads(sys.argv[2])
        out=[]
        for cfg in grid:
            ss=tuple(cfg["ss"]);sl=tuple(cfg["sl"]);pct=cfg["pct"]
            t0=time.time(); micro,per=full_score(ss,sl,nms,pct); dt=time.time()-t0
            out.append(dict(ss=ss,sl=sl,pct=pct,micro=round(micro,4),per=per,secs=round(dt,1)))
            print(f"ss={ss} sl={sl} pct={pct} micro={micro:.4f} "
                  f"clean(adj={per['44b6_0113de3b']['adj']} tp/fp/fn={per['44b6_0113de3b']['tp']}/{per['44b6_0113de3b']['fp']}/{per['44b6_0113de3b']['fn']}) "
                  f"frag(adj={per['44b6_0b24845f']['adj']} tp/fp/fn={per['44b6_0b24845f']['tp']}/{per['44b6_0b24845f']['fp']}/{per['44b6_0b24845f']['fn']} nodes={per['44b6_0b24845f']['pred_nodes']}) [{dt:.0f}s]",flush=True)
            Path("experiments/sot2272/screen_dog_full.json").write_text(json.dumps(out,indent=2))
        print("DONE")
