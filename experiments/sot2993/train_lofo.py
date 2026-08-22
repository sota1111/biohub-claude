"""SOT-2993 — train the self-trained 3D U-Net detector leave-one-family-out with
**masked sparse supervision** (the new grounds vs SOT-2828/2848/2863).

The harness is byte-identical to SOT-2863 (``experiments/sot2862a/train_lofo.py``)
— same families, same GT-centred patch sampler, same arch (``temporal_unet3d``),
same seed — so the leak-free LOFO A/B isolates the loss as the single variable. The
ONE change is the loss:

* SOT-2848 ``naive``     — foreground-weighted MSE over the WHOLE patch (background
  weight 1.0 everywhere) ⇒ every unannotated cell taught as background ⇒ PU
  contamination ⇒ degenerate (micro-adj 0.0).
* SOT-2863 ``cellsparse`` — background down-weighted to 0.05 (still supervised
  everywhere) ⇒ curbed but not cured (0.2753 << champion).
* SOT-2993 ``masked``    — loss computed ONLY inside a bounded ellipsoid mask
  around each GT annotation (:func:`biohub_tracking.learned_detect.masked_sparse_loss_weights`);
  weight is EXACTLY ZERO outside that supervised field of view, so an unannotated
  cell contributes no gradient and is never pushed toward background.

Trains four folds (holdout family trained on the other three), saving to
``experiments/sot2993/weights/masked/<family>.pt`` for ``run_ab.py`` to score
through the single leak-free CV harness. Runs in the repo ``.venv`` (torch + GPU).
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

from biohub_tracking.learned_detect import (  # noqa: E402
    _build_arch,
    masked_sparse_loss_weights,
)

FAMILIES = ("44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1")
SIGMA_ZYX = (1.0, 3.0, 3.0)     # Gaussian target half-width (voxels); matches SOT-2863
RADIUS_ZYX = (4.0, 12.0, 12.0)  # supervised-FOV ellipsoid half-axes (voxels) = 4*sigma
PATCH_ZYX = (32, 128, 128)
FG_WEIGHT = 50.0


# ---------------------------------------------------------------------------
# Harness (reused verbatim from SOT-2848/2863 so the ONLY A/B variable is the loss).
# ---------------------------------------------------------------------------
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
    """A patch centred (with jitter) on a random GT node of a random labeled frame.

    Returns ``(patch, target, weights)`` where ``weights`` are the masked-sparse
    per-voxel loss weights (0 outside the supervised FOV) — the SOT-2993 change.
    """
    t = int(rng.choice(cache.labeled_frames))
    pts = cache.gt[t]
    c = pts[rng.integers(len(pts))]
    frame = cache.frame(t)
    Z, Y, X = frame.shape
    pz, py, px = PATCH_ZYX
    z0 = int(np.clip(round(c[0]) - pz // 2 + rng.integers(-pz // 4, pz // 4 + 1), 0, max(0, Z - pz)))
    y0 = int(np.clip(round(c[1]) - py // 2 + rng.integers(-py // 4, py // 4 + 1), 0, max(0, Y - py)))
    x0 = int(np.clip(round(c[2]) - px // 2 + rng.integers(-px // 4, px // 4 + 1), 0, max(0, X - px)))
    patch = frame[z0:z0 + pz, y0:y0 + py, x0:x0 + px]
    local = []
    for q in pts:
        lz, ly, lx = q[0] - z0, q[1] - y0, q[2] - x0
        if 0 <= lz < patch.shape[0] and 0 <= ly < patch.shape[1] and 0 <= lx < patch.shape[2]:
            local.append((lz, ly, lx))
    local_pts = np.asarray(local, dtype=np.float64)
    from biohub_tracking.learned_detect import gaussian_heatmap_target
    target = gaussian_heatmap_target(patch.shape, local_pts, SIGMA_ZYX)
    weights = masked_sparse_loss_weights(
        patch.shape, local_pts, sigma_zyx=SIGMA_ZYX, radius_zyx=RADIUS_ZYX, fg_weight=FG_WEIGHT
    )
    return patch, target, weights


def masked_mse(logits, target, weights):
    """Masked sparse loss: weighted MSE reduced over the supervised FOV only.

    ``weights == 0`` outside the annotation mask ⇒ no gradient there ⇒ unannotated
    cells/background excluded from backprop (the SOT-2993 cure)."""
    import torch

    prob = torch.sigmoid(logits)
    num = (weights * (prob - target) ** 2).sum()
    den = weights.sum().clamp(min=1.0)
    return num / den


def train_fold(holdout: str, *, steps: int, batch: int, lr: float, base: int,
               seed: int, device: str, log) -> dict:
    import torch

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = torch.device(device)
    train_families = [f for f in FAMILIES if f != holdout]
    caches = [FamilyCache(f) for f in train_families]
    n_pos = sum(c.n_labels for c in caches)
    log(f"[masked|{holdout}] train on {train_families} | GT labels={n_pos} | radius={RADIUS_ZYX}")

    model = _build_arch("temporal_unet3d").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    t0 = time.time()
    losses = []
    for step in range(steps):
        patches, targets, weights = [], [], []
        for _ in range(batch):
            c = caches[rng.integers(len(caches))]
            p, tg, w = _sample_patch(c, rng)
            patches.append(p)
            targets.append(tg)
            weights.append(w)
        xb = torch.from_numpy(np.stack(patches))[:, None].to(dev)
        yb = torch.from_numpy(np.stack(targets))[:, None].to(dev)
        wb = torch.from_numpy(np.stack(weights))[:, None].to(dev)
        logits = model(xb)
        loss_val = masked_mse(logits, yb, wb)
        opt.zero_grad()
        loss_val.backward()
        opt.step()
        losses.append(float(loss_val.item()))
        if (step + 1) % max(1, steps // 10) == 0:
            log(f"[masked|{holdout}] step {step+1}/{steps} loss={np.mean(losses[-50:]):.5f}")

    dt = time.time() - t0
    out_dir = REPO / "experiments/sot2993/weights/masked"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{holdout}.pt"
    torch.save({"state_dict": model.state_dict(), "arch": "temporal_unet3d",
                "base": base, "holdout": holdout, "train_families": train_families,
                "loss": "masked", "sigma_zyx": SIGMA_ZYX, "radius_zyx": RADIUS_ZYX,
                "fg_weight": FG_WEIGHT, "steps": steps, "seed": seed},
               out_path)
    log(f"[masked|{holdout}] saved {out_path} | {dt:.1f}s final_loss={np.mean(losses[-50:]):.5f}")
    return {"holdout": holdout, "train_families": train_families, "n_train_labels": n_pos,
            "loss": "masked", "radius_zyx": RADIUS_ZYX, "final_loss": float(np.mean(losses[-50:])),
            "seconds": round(dt, 1), "weights": str(out_path)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None, help="train a single holdout fold")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    def log(m):
        print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)

    out = args.out or str(REPO / "experiments/sot2993/train_summary_masked.json")
    folds = [args.only] if args.only else list(FAMILIES)
    summary = []
    for h in folds:
        summary.append(train_fold(
            h, steps=args.steps, batch=args.batch, lr=args.lr, base=args.base,
            seed=args.seed, device=args.device, log=log))
    Path(out).write_text(json.dumps({"folds": summary, "config": vars(args)}, indent=2) + "\n")
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
