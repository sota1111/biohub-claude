"""Screen the GT-learned edge-linking cost (SOT-2841), leak-free LOFO.

The champion ``detect-link-dog-v4-shorttrack`` links each ``t -> t+1`` transition on
**scaled centroid distance alone**, so in a dense ``6bba`` family the optimal-distance
assignment attaches the wrong, merely nearer neighbour. SOT-2829 added a single
hand-crafted appearance-similarity term (non-discriminating, REJECTED); this screens
the untried **learned LINKING** axis (``label = edge``): a light logistic classifier
fit on GT *consecutive* edges (positives) vs. the feasible non-GT successors of
matched sources (confident negatives) over a joint edge-feature vector
(:data:`biohub_tracking.edge_linker.EDGE_FEATURE_NAMES`), whose learned ``p_edge``
enters the cost as ``dist + weight * (1 - p_edge)`` (re-rank only, ``<= max_distance``
gate on raw distance -> metric-valid).

**Leak-free (leave-one-family-out).** For each held-out family the model is trained
on the *other three* families' (edge-features, edge-labels) only, then applied to
link the held-out family; the four held-out family scores are aggregated into the
SOT-2817 re-anchored full-metric CV, so every family is scored by a model that never
saw it. ``weight = 0`` is the byte-invariance sanity (must reproduce the champion
0.6649 edge-for-edge). We also report the held-out positive-vs-negative ``p_edge``
gap per family as discriminative-power evidence. Detection + descriptors are computed
**once** per family and every weight variant re-links off the cache (single-variable
same-seed A/B). Writes ``experiments/sot2841/screen_edge_linker.json``. No Kaggle
submission.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.detect import detect_volume_series_with_descriptors
from biohub_tracking.edge_linker import (
    EDGE_FEATURE_NAMES,
    edge_feature_planes,
    fit_edge_cost,
)
from biohub_tracking.eval.cv import (
    CHAMPION_REFERENCE_MICRO_ADJ,
    CV_HOLDOUT,
    aggregate,
    cv_result_to_dict,
    representativeness_report,
    score_family,
)
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff
from biohub_tracking.link import LinkParams, link_centroids
from biohub_tracking.matching import match_nodes
from biohub_tracking.pipeline import _open_image_array

REPO = Path(__file__).resolve().parents[2]

# Champion per-dataset adjusted edge Jaccard — the no-regression floor (registry).
CHAMPION_PER_DATASET_ADJ = {
    "44b6_0113de3b": 0.8895,
    "44b6_0b24845f": 0.6817,
    "6bba_05b6850b": 0.5700,
    "6bba_05db0fb1": 0.7310,
}

# Re-rank strength sweep (microns of penalty for a p_edge=0 successor).
# 0.0 == champion byte-invariance sanity; the rest probe increasing learned pull.
WEIGHTS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

MAX_DISTANCE = 7.0


def _pred_graph_for_labels(dets, scale):
    """A distance-only linked graph, used only to id predicted nodes per timepoint.

    We need pred node ids/timepoints to match against GT; the champion linking is
    irrelevant to labelling (labels are per feasible pair, computed directly), so a
    cheap default link suffices to get a :class:`TrackingGraph` with the same node
    layout the linker uses (timepoint order, then detection order).
    """
    return link_centroids(dets, scale=scale, params=LinkParams(max_distance=MAX_DISTANCE))


def build_family_training(fam_name, dets, descs, gt, scale):
    """Edge-features + 0/1 labels for every trainable feasible pair of one family.

    A feasible pair ``(i @ t) -> (j @ t+1)`` (scaled distance ``<= MAX_DISTANCE``) is:
      * **positive** iff pred ``i`` matches GT node ``u``, pred ``j`` matches GT node
        ``v``, and ``(u, v)`` is a GT edge;
      * **negative** iff pred ``i`` matches some GT node (source is a real cell, so a
        definite GT successor exists) but the pair is not that GT edge;
      * **skipped** iff pred ``i`` is unmatched (PU-ambiguous source — excluded to
        avoid the SOT-2828 node-PU contamination).
    Returns ``(features (N, F), labels (N,))``.
    """
    scale_arr = np.asarray(scale, dtype=float)
    pred = _pred_graph_for_labels(dets, scale)
    # pred_id -> gt_id (per-timepoint <=7um optimal matching, the competition rule).
    p2g = match_nodes(pred, gt, scale=tuple(scale_arr), max_distance=MAX_DISTANCE)

    # Node ids are assigned densely: timepoint order, then detection order. Rebuild
    # the (t, det_index) -> node_id map so we can look up matches per feasible pair.
    ids_by_t: dict[int, list[int]] = {}
    nid = 0
    for t in sorted(dets):
        ids_by_t[t] = list(range(nid, nid + len(dets[t])))
        nid += len(dets[t])

    gt_edges = set(gt.edges)

    feats: list[np.ndarray] = []
    labels: list[float] = []
    times = sorted(dets)
    for t_a, t_b in zip(times, times[1:]):
        if t_b != t_a + 1:
            continue
        src, dst = dets[t_a], dets[t_b]
        if len(src) == 0 or len(dst) == 0:
            continue
        diff = (src[:, None, :] - dst[None, :, :]) * scale_arr
        dist = np.sqrt((diff**2).sum(axis=2))
        planes = edge_feature_planes(dist, descs[t_a], descs[t_b], MAX_DISTANCE)
        src_ids, dst_ids = ids_by_t[t_a], ids_by_t[t_b]
        fi, fj = np.where(dist <= MAX_DISTANCE)
        for i, j in zip(fi.tolist(), fj.tolist()):
            gu = p2g.get(src_ids[i])
            if gu is None:
                continue  # unmatched source: PU-ambiguous, skip
            gv = p2g.get(dst_ids[j])
            is_pos = gv is not None and (gu, gv) in gt_edges
            feats.append(planes[i, j])
            labels.append(1.0 if is_pos else 0.0)
    if not feats:
        return np.zeros((0, len(EDGE_FEATURE_NAMES))), np.zeros(0)
    return np.asarray(feats, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def cv_for_weight(cache, training, weight):
    """LOFO CV at a given re-rank strength + per-family held-out p_edge gap."""
    rows = []
    gaps = {}
    for fam in CV_HOLDOUT:
        dets, descs, gt, scale, n_true = cache[fam.name]
        # Train on the OTHER three families only (leak-free).
        Xs, ys = [], []
        for other in CV_HOLDOUT:
            if other.name == fam.name:
                continue
            X, y = training[other.name]
            if len(X):
                Xs.append(X)
                ys.append(y)
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        model = fit_edge_cost(X, y, weight=weight)
        link = LinkParams(
            max_distance=MAX_DISTANCE,
            allow_division=False,
            division_distance=MAX_DISTANCE,
            min_track_length=4,
            edge_cost_model=model.to_dict(),
        )
        pred = link_centroids(dets, scale=scale, params=link, descriptors=descs)
        rows.append(score_family(fam, pred, gt, n_true, scale=scale))
        # Held-out discriminative gap on THIS family's own labels (model never saw it).
        Xf, yf = training[fam.name]
        if len(Xf) and yf.sum() > 0 and (len(yf) - yf.sum()) > 0:
            p = model.probability_rows(Xf)
            gaps[fam.name] = {
                "p_pos_mean": round(float(p[yf > 0.5].mean()), 4),
                "p_neg_mean": round(float(p[yf < 0.5].mean()), 4),
                "gap": round(float(p[yf > 0.5].mean() - p[yf < 0.5].mean()), 4),
                "n_pos": int(yf.sum()),
                "n_neg": int(len(yf) - yf.sum()),
            }
    return aggregate(rows), gaps


def edge_counts_by_dataset(res) -> dict:
    return {
        r.name: {"tp": r.edge_tp, "fp": r.edge_fp, "fn": r.edge_fn,
                 "pred_nodes": r.num_pred_nodes}
        for r in res.per_dataset
    }


def summarise(res, baseline, base_ec, weight, gaps):
    per = {r.name: r.adj_edge_jaccard for r in res.per_dataset}
    no_reg = res.no_regression_vs(CHAMPION_PER_DATASET_ADJ)
    score_up = res.score > baseline.score + 1e-9
    ec = edge_counts_by_dataset(res)
    edge_delta = {
        name: {
            "d_tp": ec[name]["tp"] - base_ec[name]["tp"],
            "d_fp": ec[name]["fp"] - base_ec[name]["fp"],
            "d_fn": ec[name]["fn"] - base_ec[name]["fn"],
            "d_pred_nodes": ec[name]["pred_nodes"] - base_ec[name]["pred_nodes"],
        }
        for name in ec
    }
    tot = {
        "d_tp": sum(v["d_tp"] for v in edge_delta.values()),
        "d_fp": sum(v["d_fp"] for v in edge_delta.values()),
        "d_fn": sum(v["d_fn"] for v in edge_delta.values()),
    }
    rep = representativeness_report(res)
    return {
        "weight": weight,
        "micro_adj_edge_jaccard": round(res.micro_adj_edge_jaccard, 4),
        "micro_edge_jaccard": round(res.micro_edge_jaccard, 4),
        "score": round(res.score, 4),
        "delta_score_vs_champion": round(res.score - baseline.score, 4),
        "no_per_dataset_regression": bool(no_reg),
        "promotable": bool(no_reg and score_up),
        "family_mix_sensitive": bool(rep.get("family_mix_sensitive")),
        "micro_lineage_macro_gap": rep.get("micro_lineage_macro_gap"),
        "total_edge_delta": tot,
        "per_dataset_adj": {k: round(v, 4) for k, v in per.items()},
        "edge_delta_by_dataset": edge_delta,
        "held_out_prob_gap": gaps,
        "cv": cv_result_to_dict(res),
    }


def main() -> int:
    cfg = load_champion_config()
    detect, base_link, cfg_scale = champion_params(cfg)

    cache: dict = {}
    training: dict = {}
    for fam in CV_HOLDOUT:
        t0 = time.time()
        arr = _open_image_array(REPO / fam.image)
        gt = load_geff(REPO / fam.geff)
        scale = geff_scale(REPO / fam.geff)
        n_true = geff_estimated_num_nodes(REPO / fam.geff)
        dets, descs = detect_volume_series_with_descriptors(arr, detect, scale=scale)
        cache[fam.name] = (dets, descs, gt, scale, n_true)
        X, y = build_family_training(fam.name, dets, descs, gt, scale)
        training[fam.name] = (X, y)
        print(f"[detect] {fam.name}: {time.time()-t0:.1f}s "
              f"dets={sum(len(v) for v in dets.values())} "
              f"train_pairs={len(y)} pos={int(y.sum())}", flush=True)

    # Champion baseline (distance-only) through the same CV.
    base_rows = []
    for fam in CV_HOLDOUT:
        dets, descs, gt, scale, n_true = cache[fam.name]
        pred = link_centroids(dets, scale=scale, params=base_link)
        base_rows.append(score_family(fam, pred, gt, n_true, scale=scale))
    baseline = aggregate(base_rows)
    base_ec = edge_counts_by_dataset(baseline)
    print(f"[baseline] champion micro_adj={baseline.micro_adj_edge_jaccard:.4f} "
          f"score={baseline.score:.4f}", flush=True)

    variants = []
    for weight in WEIGHTS:
        res, gaps = cv_for_weight(cache, training, weight)
        row = summarise(res, baseline, base_ec, weight, gaps)
        variants.append(row)
        print(f"[weight] w={weight:.1f} micro_adj={res.micro_adj_edge_jaccard:.4f} "
              f"score={res.score:.4f} d={row['delta_score_vs_champion']:+.4f} "
              f"dTP={row['total_edge_delta']['d_tp']:+d} "
              f"dFP={row['total_edge_delta']['d_fp']:+d} "
              f"dFN={row['total_edge_delta']['d_fn']:+d} "
              f"no_reg={row['no_per_dataset_regression']} "
              f"promotable={row['promotable']}", flush=True)

    zero_row = next(v for v in variants if v["weight"] == 0.0)
    sanity_reproduces_champion = (
        abs(zero_row["score"] - round(baseline.score, 4)) < 1e-9
        and zero_row["total_edge_delta"] == {"d_tp": 0, "d_fp": 0, "d_fn": 0}
    )

    variants_ranked = sorted(variants, key=lambda v: v["score"], reverse=True)
    promotable = [v for v in variants_ranked if v["promotable"]]

    # Mean held-out discriminative gap (positive over the whole sweep, weight-invariant
    # since the model coefficients depend only on the LOFO training, not on `weight`).
    ref = next(v for v in variants if v["weight"] == 1.0)
    mean_gap = None
    if ref["held_out_prob_gap"]:
        gvals = [g["gap"] for g in ref["held_out_prob_gap"].values()]
        mean_gap = round(sum(gvals) / len(gvals), 4)

    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2841",
        "axis": "GT-learned edge-linking cost (label=edge): leave-one-family-out "
                "logistic over [dist, appearance-cos, src/dst rival counts, succ/pred "
                "distance ranks, succ margin]; link cost dist + w*(1-p_edge), "
                "<=max_distance gate on raw distance (metric-valid, re-rank only), "
                "single-variable same-seed A/B vs the frozen "
                "detect-link-dog-v4-shorttrack champion",
        "cv_source": "biohub_tracking.eval.cv (SOT-2817 re-anchored full-metric "
                     "4-family leak-free holdout)",
        "leak_free_protocol": "leave-one-family-out: each held-out family is linked "
                              "by a model trained ONLY on the other three families' "
                              "edge (features,labels); positives=GT consecutive edges, "
                              "negatives=feasible non-GT successors of matched sources, "
                              "unmatched-source pairs excluded (no node PU).",
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "feature_names": list(EDGE_FEATURE_NAMES),
        "baseline_champion": {
            "micro_adj_edge_jaccard": round(baseline.micro_adj_edge_jaccard, 4),
            "score": round(baseline.score, 4),
            "edge_counts_by_dataset": base_ec,
            "cv": cv_result_to_dict(baseline),
        },
        "champion_per_dataset_adj_floor": CHAMPION_PER_DATASET_ADJ,
        "grid": {"weight": WEIGHTS},
        "weight_zero_sanity_reproduces_champion": bool(sanity_reproduces_champion),
        "mean_held_out_prob_gap_at_w1": mean_gap,
        "variants_ranked_by_score": variants_ranked,
        "n_promotable": len(promotable),
        "best_promotable": promotable[0] if promotable else None,
    }
    out = REPO / "experiments/sot2841/screen_edge_linker.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"baseline champion score={baseline.score:.4f} "
          f"(micro_adj={baseline.micro_adj_edge_jaccard:.4f})")
    print(f"weight=0 reproduces champion exactly: {sanity_reproduces_champion}")
    print(f"mean held-out p_edge gap (w=1): {mean_gap}")
    print(f"n_promotable={len(promotable)}")
    if promotable:
        b = promotable[0]
        print(f"BEST PROMOTABLE: w={b['weight']} score={b['score']} "
              f"(+{b['delta_score_vs_champion']}) edge_delta={b['total_edge_delta']}")
    else:
        top = variants_ranked[0]
        print(f"NO PROMOTABLE variant. Top-by-score: w={top['weight']} "
              f"score={top['score']} (delta {top['delta_score_vs_champion']:+}) "
              f"no_reg={top['no_per_dataset_regression']} "
              f"edge_delta={top['total_edge_delta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
