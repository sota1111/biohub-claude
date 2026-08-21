"""SOT-2863 (cycle-9 child A) — train the learned detector leave-one-family-out
with a **PU-aware loss** (nnPU / Cellsparse) to cure the sparse-GT contamination
that made SOT-2848's naive-supervised detector degenerate.

Why a different loss (root cause, not a blind retry)
----------------------------------------------------
SOT-2848 trained ``temporal_unet3d`` with a foreground-weighted **MSE** on a
Gaussian-blob heatmap: every voxel NOT within a labeled blob is a regression
target of 0. But the competition train split ships **sparse** tracking GT (~1
labeled cell/frame for the two 44b6 videos, ~9-12 for the two 6bba videos), so
the vast majority of genuine cell voxels are *unlabeled* and the naive loss
pushes them to background → **Positive-Unlabeled (PU) contamination** → no
family-robust operating point (measured micro-adj 0.0 at thr 0.5, wild per-family
variance at thr 0.3). This module keeps SOT-2848's harness **byte-identical**
(same families, same patch sampler, same seed, same arch) and swaps ONLY the
training loss, so the leak-free LOFO A/B isolates the loss as the single variable:

* ``naive``     — SOT-2848's foreground-weighted MSE (reproduces the baseline arm).
* ``nnpu``      — non-negative PU risk (Kiryo et al., arXiv:1703.00593;
  ``kiryor/nnPUlearning/pu_loss.py``; PU cell detection arXiv:2106.15918). Treats
  blob-core voxels as **labeled Positives** and everything else as **Unlabeled**,
  with the non-negative clamp on the empirical negative risk that stops the model
  overfitting unlabeled true-positives as negatives. ``pi`` (cell-voxel prior) is
  estimated from GT node density (``--pi`` overrides).
* ``cellsparse`` — Cellsparse-style **ignore-weighting** (biorxiv 2023.06.13.544786):
  the naive MSE but the unlabeled (non-blob) voxel loss is down-weighted to
  ``--unlabeled-weight`` (~0.05) so sparse background barely supervises.

Trains four folds: for each holdout family a model is trained on the *other
three* families' sparse GT and saved to ``weights/<loss>/<family>.pt``, to be
scored on the held-out family by ``run_ab.py`` through the single leak-free CV
harness (``biohub_tracking.eval.cv``). Runs in the repo ``.venv`` (torch + GPU).
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

from biohub_tracking.io import geff_estimated_num_nodes  # noqa: E402
from biohub_tracking.learned_detect import _build_arch  # noqa: E402

FAMILIES = ("44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1")
SIGMA_ZYX = (1.0, 3.0, 3.0)  # Gaussian target half-width (voxels), anisotropic
PATCH_ZYX = (32, 128, 128)
POS_CORE = 0.5  # heatmap target >= this is a labeled Positive voxel (nnPU/cellsparse)


# ---------------------------------------------------------------------------
# Harness (reused verbatim from SOT-2848 so the ONLY A/B variable is the loss).
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
    z0 = int(np.clip(round(c[0]) - pz // 2 + rng.integers(-pz // 4, pz // 4 + 1), 0, max(0, Z - pz)))
    y0 = int(np.clip(round(c[1]) - py // 2 + rng.integers(-py // 4, py // 4 + 1), 0, max(0, Y - py)))
    x0 = int(np.clip(round(c[2]) - px // 2 + rng.integers(-px // 4, px // 4 + 1), 0, max(0, X - px)))
    patch = frame[z0:z0 + pz, y0:y0 + py, x0:x0 + px]
    local = []
    for q in pts:
        lz, ly, lx = q[0] - z0, q[1] - y0, q[2] - x0
        if 0 <= lz < patch.shape[0] and 0 <= ly < patch.shape[1] and 0 <= lx < patch.shape[2]:
            local.append((lz, ly, lx))
    target = _gaussian_target(patch.shape, np.asarray(local, dtype=np.float64))
    return patch, target


# ---------------------------------------------------------------------------
# PU-aware losses (the single A/B variable vs SOT-2848).
# ---------------------------------------------------------------------------
def estimate_pi(caches: list[FamilyCache], source: str = "estimated") -> float:
    """Estimate the cell-voxel prior ``pi`` from GT node density.

    ``pi`` = (cells per frame x voxels in one blob core) / voxels per volume,
    averaged over the training families. One blob-core voxel count is measured
    directly from a single :func:`_gaussian_target` peak (target >= ``POS_CORE``).

    Two node-density sources (``--pi-source``):

    * ``labeled`` — cells/frame from the **sparse .geff labels** the model is
      supervised on. This is what the issue literally names ("GT node density"),
      but the split is heavily sub-sampled (~1-12 labeled cells/frame vs a metric
      true-count estimate of 25k-70k nodes/family), so it is a severe *undercount*
      of the true prior and drives nnPU into degenerate under-firing (the positive
      risk term vanishes).
    * ``estimated`` (default) — cells/frame from the harness's **estimated true
      node count** (``geff_estimated_num_nodes``, the same true-count the metric's
      node-count penalty uses), i.e. the realistic cell-voxel prevalence nnPU's
      non-negative correction is designed to exploit. This is the principled
      reading of "cell-voxel 割合" and the primary configuration; ``labeled`` is
      run as a diagnostic. Override either with ``--pi``.
    """
    Z, Y, X = PATCH_ZYX
    one = _gaussian_target((Z, Y, X), np.asarray([[Z // 2, Y // 2, X // 2]], dtype=np.float64))
    core_voxels = int((one >= POS_CORE).sum())
    fracs = []
    for c in caches:
        vox_per_vol = int(np.prod(c.vol.shape[1:]))  # Z*Y*X
        if source == "labeled":
            cells_per_frame = c.n_labels / max(1, len(c.labeled_frames))
        else:  # estimated true-count node density
            true_nodes = geff_estimated_num_nodes(REPO / f"data/train/{c.name}.geff")
            cells_per_frame = true_nodes / max(1, c.T)
        fracs.append(cells_per_frame * core_voxels / vox_per_vol)
    return float(np.mean(fracs))


def compute_loss(logits, target, *, loss: str, pi: float, fg_weight: float,
                 unlabeled_weight: float, beta: float, gamma: float):
    """Return the scalar training loss for a batch of logits/targets.

    ``naive``/``cellsparse`` regress a sigmoid heatmap (MSE); ``nnpu`` treats the
    detector as a per-voxel PU classifier on the raw logits.
    """
    import torch
    import torch.nn.functional as F

    if loss in ("naive", "cellsparse"):
        prob = torch.sigmoid(logits)
        pos = (target > 0.05).float()
        if loss == "naive":
            # foreground-weighted MSE (SOT-2848): background weight 1.0
            w = 1.0 + (fg_weight - 1.0) * pos
        else:
            # Cellsparse ignore-weighting: unlabeled (background) down-weighted
            w = unlabeled_weight + (fg_weight - unlabeled_weight) * pos
        return (w * (prob - target) ** 2).mean()

    if loss == "nnpu":
        # Per-voxel PU classification on logits. Labeled Positives = blob cores;
        # everything else is Unlabeled. Logistic surrogate: l_+(z)=softplus(-z),
        # l_-(z)=softplus(z). nnPU risk = pi*R_P+ + max_beta(R_U- - pi*R_P-),
        # with the Kiryo non-negative correction (gradient ascent when the
        # empirical negative risk dips below -beta).
        p_mask = (target >= POS_CORE)
        u_mask = ~p_mask
        l_pos = F.softplus(-logits)   # loss if voxel is positive
        l_neg = F.softplus(logits)    # loss if voxel is negative
        n_p = p_mask.sum()
        if n_p == 0:
            # no labeled positive in this batch: fall back to unlabeled-as-neg risk
            return l_neg[u_mask].mean()
        r_p_pos = l_pos[p_mask].mean()
        r_p_neg = l_neg[p_mask].mean()
        r_u_neg = l_neg[u_mask].mean() if u_mask.any() else torch.zeros((), device=logits.device)
        pos_risk = pi * r_p_pos
        neg_risk = r_u_neg - pi * r_p_neg
        if float(neg_risk.item()) >= -beta:
            return pos_risk + neg_risk
        # non-negative correction: ascend to push the negative risk back up
        return -gamma * neg_risk

    raise ValueError(f"unknown loss {loss!r}")


def train_fold(holdout: str, *, loss: str, steps: int, batch: int, lr: float, base: int,
               fg_weight: float, unlabeled_weight: float, pi_override: float | None,
               pi_source: str, beta: float, gamma: float, seed: int, device: str, log) -> dict:
    import torch

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = torch.device(device)
    train_families = [f for f in FAMILIES if f != holdout]
    caches = [FamilyCache(f) for f in train_families]
    n_pos = sum(c.n_labels for c in caches)
    pi = pi_override if pi_override is not None else estimate_pi(caches, pi_source)
    log(f"[{loss}|{holdout}] train on {train_families} | GT labels={n_pos} | pi={pi:.3e}")

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
        loss_val = compute_loss(logits, yb, loss=loss, pi=pi, fg_weight=fg_weight,
                                unlabeled_weight=unlabeled_weight, beta=beta, gamma=gamma)
        opt.zero_grad()
        loss_val.backward()
        opt.step()
        losses.append(float(loss_val.item()))
        if (step + 1) % max(1, steps // 10) == 0:
            log(f"[{loss}|{holdout}] step {step+1}/{steps} loss={np.mean(losses[-50:]):.5f}")

    dt = time.time() - t0
    out_dir = REPO / f"experiments/sot2862a/weights/{loss}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{holdout}.pt"
    torch.save({"state_dict": model.state_dict(), "arch": "temporal_unet3d",
                "base": base, "holdout": holdout, "train_families": train_families,
                "loss": loss, "pi": pi, "sigma_zyx": SIGMA_ZYX, "steps": steps, "seed": seed},
               out_path)
    log(f"[{loss}|{holdout}] saved {out_path} | {dt:.1f}s final_loss={np.mean(losses[-50:]):.5f}")
    return {"holdout": holdout, "train_families": train_families, "n_train_labels": n_pos,
            "loss": loss, "pi": pi, "final_loss": float(np.mean(losses[-50:])),
            "seconds": round(dt, 1), "weights": str(out_path)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["naive", "nnpu", "cellsparse"], required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--fg-weight", type=float, default=50.0)
    ap.add_argument("--unlabeled-weight", type=float, default=0.05,
                    help="Cellsparse: background/unlabeled voxel loss weight")
    ap.add_argument("--pi", type=float, default=None, help="override nnPU cell-voxel prior")
    ap.add_argument("--pi-source", choices=["estimated", "labeled"], default="estimated",
                    help="nnPU pi from estimated-true node density (default) or sparse labels")
    ap.add_argument("--beta", type=float, default=0.0, help="nnPU non-negative slack")
    ap.add_argument("--gamma", type=float, default=1.0, help="nnPU ascent factor")
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

    out = args.out or str(REPO / f"experiments/sot2862a/train_summary_{args.loss}.json")
    folds = [args.only] if args.only else list(FAMILIES)
    summary = []
    for h in folds:
        summary.append(train_fold(
            h, loss=args.loss, steps=args.steps, batch=args.batch, lr=args.lr,
            base=args.base, fg_weight=args.fg_weight, unlabeled_weight=args.unlabeled_weight,
            pi_override=args.pi, pi_source=args.pi_source, beta=args.beta, gamma=args.gamma,
            seed=args.seed, device=args.device, log=log))
    Path(out).write_text(json.dumps({"folds": summary, "config": vars(args)}, indent=2) + "\n")
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
