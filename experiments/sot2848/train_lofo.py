"""SOT-2848 — train the learned TemporalUNet3D detector leave-one-family-out.

Leak-free learned-detector CV needs the scored video to be **absent from the
detector's training set** (unlike the classical champion, which has no learned
parameters and cannot leak). So we train four folds: for each of the four
holdout families we train a ``temporal_unet3d`` (registered in
``biohub_tracking.learned_detect``) on the *other three* families' sparse GT and
save its weights, to be scored on the held-out family by ``run_ab.py``.

Supervision (sparse, PU-aware)
------------------------------
The competition train split ships **sparse tracking GT** — only a few tracked
cells per frame (measured: ~1 labeled cell/frame for the two 44b6 videos, ~9-12
for the two 6bba videos), NOT dense cell masks. We build a Gaussian-blob target
at each GT node (voxel coords) and train patches centred on GT positives with a
foreground-weighted MSE on the sigmoid heatmap. Input normalisation is the SAME
per-volume standardisation the receptacle applies at inference
(``LearnedDetector._heatmap``), so training and inference see identical inputs.

Runs in the repo ``.venv`` (torch is installed there for SOT-2848); GPU auto.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from biohub_tracking.learned_detect import _build_arch  # noqa: E402

FAMILIES = ("44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1")
SIGMA_ZYX = (1.0, 3.0, 3.0)  # Gaussian target half-width (voxels), anisotropic
PATCH_ZYX = (32, 128, 128)


def _load_volume(name: str):
    arr = zarr.open(str(REPO / f"data/test/{name}.zarr"), mode="r")["0"]
    return arr  # lazy (T, Z, Y, X) uint16


def _load_gt_points(name: str) -> dict[int, np.ndarray]:
    r = zarr.open(str(REPO / f"data/train/{name}.geff"), mode="r")
    t = r["nodes/props/t/values"][:].astype(np.int64)
    z = r["nodes/props/z/values"][:].astype(np.float64)
    y = r["nodes/props/y/values"][:].astype(np.float64)
    x = r["nodes/props/x/values"][:].astype(np.float64)
    out: dict[int, list] = {}
    for tt, zz, yy, xx in zip(t, z, y, x):
        out.setdefault(int(tt), []).append((zz, yy, xx))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def _standardize(vol: np.ndarray) -> np.ndarray:
    """Match LearnedDetector._heatmap: per-volume (v-mean)/std standardisation."""
    v = np.asarray(vol, dtype=np.float32)
    mean = float(v.mean())
    std = float(v.std()) or 1.0
    return (v - mean) / std


def _gaussian_target(shape, points_zyx: np.ndarray, sigma=SIGMA_ZYX) -> np.ndarray:
    """Sum-of-Gaussians heatmap (peak 1.0) at each point in a (Z,Y,X) patch."""
    target = np.zeros(shape, dtype=np.float32)
    if points_zyx.size == 0:
        return target
    sz, sy, sx = sigma
    rz, ry, rx = int(3 * sz) + 1, int(3 * sy) + 1, int(3 * sx) + 1
    Z, Y, X = shape
    for pz, py, px in points_zyx:
        pz, py, px = int(round(pz)), int(round(py)), int(round(px))
        z0, z1 = max(0, pz - rz), min(Z, pz + rz + 1)
        y0, y1 = max(0, py - ry), min(Y, py + ry + 1)
        x0, x1 = max(0, px - rx), min(X, px + rx + 1)
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue
        zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
        g = np.exp(
            -(((zz - pz) ** 2) / (2 * sz**2)
              + ((yy - py) ** 2) / (2 * sy**2)
              + ((xx - px) ** 2) / (2 * sx**2))
        )
        np.maximum(target[z0:z1, y0:y1, x0:x1], g.astype(np.float32), out=target[z0:z1, y0:y1, x0:x1])
    return target


class FamilyCache:
    """Standardised full-volume frames + GT points, loaded lazily per frame."""

    def __init__(self, name: str):
        self.name = name
        self.vol = _load_volume(name)
        self.T = self.vol.shape[0]
        self.gt = _load_gt_points(name)
        self.labeled_frames = sorted(self.gt)
        self.n_labels = sum(len(v) for v in self.gt.values())

    def frame(self, t: int) -> np.ndarray:
        return _standardize(self.vol[t][:])  # (Z,Y,X) float32


def _sample_patch(cache: FamilyCache, rng: np.random.Generator):
    """A patch centred (with jitter) on a random GT node of a random labeled frame."""
    t = int(rng.choice(cache.labeled_frames))
    pts = cache.gt[t]
    c = pts[rng.integers(len(pts))]
    frame = cache.frame(t)
    Z, Y, X = frame.shape
    pz, py, px = PATCH_ZYX
    # top-left with jitter so the node is not always centred
    z0 = int(np.clip(round(c[0]) - pz // 2 + rng.integers(-pz // 4, pz // 4 + 1), 0, max(0, Z - pz)))
    y0 = int(np.clip(round(c[1]) - py // 2 + rng.integers(-py // 4, py // 4 + 1), 0, max(0, Y - py)))
    x0 = int(np.clip(round(c[2]) - px // 2 + rng.integers(-px // 4, px // 4 + 1), 0, max(0, X - px)))
    patch = frame[z0:z0 + pz, y0:y0 + py, x0:x0 + px]
    # local GT points inside the patch window
    local = []
    for q in pts:
        lz, ly, lx = q[0] - z0, q[1] - y0, q[2] - x0
        if 0 <= lz < patch.shape[0] and 0 <= ly < patch.shape[1] and 0 <= lx < patch.shape[2]:
            local.append((lz, ly, lx))
    target = _gaussian_target(patch.shape, np.asarray(local, dtype=np.float64))
    return patch, target


def train_fold(holdout: str, *, steps: int, batch: int, lr: float, base: int,
               fg_weight: float, seed: int, device: str, log) -> dict:
    import torch

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = torch.device(device)
    train_families = [f for f in FAMILIES if f != holdout]
    caches = [FamilyCache(f) for f in train_families]
    n_pos = sum(c.n_labels for c in caches)
    log(f"[fold {holdout}] train on {train_families} | total GT labels={n_pos}")

    model = _build_arch("temporal_unet3d").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    t0 = time.time()
    losses = []
    for step in range(steps):
        patches, targets = [], []
        for _ in range(batch):
            c = caches[rng.integers(len(caches))]
            p, tg = _sample_patch(c, rng)
            patches.append(p)
            targets.append(tg)
        xb = torch.from_numpy(np.stack(patches))[:, None].to(dev)
        yb = torch.from_numpy(np.stack(targets))[:, None].to(dev)
        logits = model(xb)
        prob = torch.sigmoid(logits)
        # foreground-weighted MSE: weight positive (target>0.05) voxels up to
        # counter the extreme background dominance of sparse point supervision.
        w = 1.0 + (fg_weight - 1.0) * (yb > 0.05).float()
        loss = (w * (prob - yb) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % max(1, steps // 10) == 0:
            log(f"[fold {holdout}] step {step+1}/{steps} loss={np.mean(losses[-50:]):.5f}")

    dt = time.time() - t0
    out_path = REPO / f"experiments/sot2848/weights/{holdout}.pt"
    torch.save({"state_dict": model.state_dict(), "arch": "temporal_unet3d",
                "base": base, "holdout": holdout, "train_families": train_families,
                "sigma_zyx": SIGMA_ZYX, "steps": steps, "seed": seed}, out_path)
    log(f"[fold {holdout}] saved {out_path} | {dt:.1f}s final_loss={np.mean(losses[-50:]):.5f}")
    return {"holdout": holdout, "train_families": train_families, "n_train_labels": n_pos,
            "final_loss": float(np.mean(losses[-50:])), "seconds": round(dt, 1),
            "weights": str(out_path)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--fg-weight", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None, help="train a single holdout fold")
    ap.add_argument("--out", default=str(REPO / "experiments/sot2848/train_summary.json"))
    args = ap.parse_args(argv)

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    def log(m):
        print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)

    folds = [args.only] if args.only else list(FAMILIES)
    summary = []
    for h in folds:
        summary.append(train_fold(h, steps=args.steps, batch=args.batch, lr=args.lr,
                                   base=args.base, fg_weight=args.fg_weight,
                                   seed=args.seed, device=args.device, log=log))
    Path(args.out).write_text(json.dumps({"folds": summary, "config": vars(args)}, indent=2) + "\n")
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
