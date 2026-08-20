"""Train the portable GT-learned detection scorer (SOT-2828).

Fits the pure-numpy logistic :class:`biohub_tracking.detect_scorer.LearnedScorer`
from the sparse GT of the four leak-free CV families and writes its embeddable
config block (``feature_names`` / ``mean`` / ``std`` / ``coef`` / ``intercept`` /
``select``). The coefficients are the *only* thing a promoted champion needs — the
kernel runs inference with numpy alone (no sklearn, no pickle, no ``data/``).

Usage (from the repo root, with the competition ``data/`` volumes present)::

    .venv/bin/python scripts/train_detect_scorer.py --out champion/detect_scorer.json --select 0.5

This trains on **all four** families (the whole holdout = the LB target, the same
methodology the champion params were fit under). For a *leak-free* estimate of the
gain use the leave-one-family-out screen instead
(``experiments/sot2828/screen_detect_scorer.py``); this script produces the artifact
to embed only *after* that screen shows a non-regressing promotion.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.detect import _normalize_intensity, detect_centroids
from biohub_tracking.detect_scorer import (
    extract_candidate_features,
    fit_scorer,
    label_candidates,
)
from biohub_tracking.eval.cv import CV_HOLDOUT
from biohub_tracking.io import geff_scale, load_geff
from biohub_tracking.pipeline import _open_image_array

REPO = Path(__file__).resolve().parents[1]


def collect(detect_params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool candidate features + sparse-GT labels + PU weights over all families."""
    feats_all, labels_all, weights_all = [], [], []
    for fam in CV_HOLDOUT:
        t0 = time.time()
        arr = _open_image_array(REPO / fam.image)
        gt = load_geff(REPO / fam.geff)
        scale = geff_scale(REPO / fam.geff)
        dets_by_t: dict[int, np.ndarray] = {}
        feat_rows = []
        for t in range(arr.shape[0]):
            vol = np.asarray(arr[t])
            coords = detect_centroids(vol, detect_params)
            dets_by_t[t] = coords
            if coords.shape[0]:
                vol_norm = _normalize_intensity(
                    vol.astype(np.float32), detect_params.intensity_norm
                )
                feats, _ = extract_candidate_features(vol_norm, coords, detect_params)
                feat_rows.append(feats)
        labels, weights = label_candidates(dets_by_t, gt, scale)
        if feat_rows:
            feats_all.append(np.concatenate(feat_rows, axis=0))
            labels_all.append(labels)
            weights_all.append(weights)
        print(f"[collect] {fam.name}: {time.time()-t0:.1f}s pos={int(labels.sum())}",
              flush=True)
    return (
        np.concatenate(feats_all, axis=0),
        np.concatenate(labels_all, axis=0),
        np.concatenate(weights_all, axis=0),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the SOT-2828 detection scorer.")
    ap.add_argument("--out", type=str, default="champion/detect_scorer.json")
    ap.add_argument("--select", type=float, default=0.5,
                    help="Keep-probability threshold embedded in the scorer.")
    ap.add_argument("--l2", type=float, default=0.1)
    args = ap.parse_args(argv)

    cfg = load_champion_config()
    detect_params, _link, _scale = champion_params(cfg)
    feats, labels, weights = collect(detect_params)
    print(f"[train] N={len(labels)} pos_frac={labels.mean():.4f}", flush=True)

    scorer = fit_scorer(feats, labels, weights, l2=args.l2, select_value=args.select)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scorer.to_dict(), indent=2) + "\n")
    print(f"wrote {out}")
    print("coef:", [round(float(v), 3) for v in scorer.coef])
    print("intercept:", round(float(scorer.intercept), 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
