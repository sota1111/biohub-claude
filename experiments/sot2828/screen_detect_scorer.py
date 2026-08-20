"""Screen the GT-learned portable detection scorer (SOT-2828).

The champion ``detect-link-dog-v4-shorttrack`` keeps NMS candidates by an
unsupervised robust threshold (``mad_k=3.0``). Every *global* operating-point lever
on top of it was REJECTED (cycles 2-5). This screens the untried **supervised**
axis: a pure-numpy logistic scorer learned from the sparse GT re-ranks / selects the
candidates to cut FP without a new global threshold.

**Leak-free evaluation = leave-one-family-out (LOFO).** A learned model scored on
the same family it was fit on would leak, so each of the four holdout families is
scored with a scorer trained on the **other three** families only. The per-family
held-out predictions are aggregated through the SOT-2817 re-anchored full-metric CV,
so the number is byte-comparable to the registry champion (0.6649). A probability
**threshold sweep** trades recall (keep dim TP) against precision (drop FP); each
threshold is a same-seed A/B (detection + features cached once, only the keep-mask
changes). ``threshold=0.0`` keeps every candidate ⇒ must reproduce the champion
byte-for-byte (the byte-invariance sanity). Writes
``experiments/sot2828/screen_detect_scorer.json``. No Kaggle submission.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.detect import _normalize_intensity, detect_centroids
from biohub_tracking.detect_scorer import (
    extract_candidate_features,
    fit_scorer,
    label_candidates,
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
from biohub_tracking.link import link_centroids
from biohub_tracking.pipeline import _open_image_array

REPO = Path(__file__).resolve().parents[2]

CHAMPION_PER_DATASET_ADJ = {
    "44b6_0113de3b": 0.8895,
    "44b6_0b24845f": 0.6817,
    "6bba_05b6850b": 0.5700,
    "6bba_05db0fb1": 0.7310,
}

# Probability keep-thresholds. 0.0 keeps every candidate (byte-invariance sanity);
# the rest probe increasingly aggressive FP pruning.
THRESHOLDS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def detect_and_features(detect_params):
    """Per family: detect per timepoint, extract candidate features + sparse-GT
    labels. Returns a cache keyed by family name."""
    cache: dict = {}
    for fam in CV_HOLDOUT:
        t0 = time.time()
        arr = _open_image_array(REPO / fam.image)
        gt = load_geff(REPO / fam.geff)
        scale = geff_scale(REPO / fam.geff)
        n_true = geff_estimated_num_nodes(REPO / fam.geff)

        n_t = arr.shape[0]
        dets_by_t: dict[int, np.ndarray] = {}
        feat_rows: list[np.ndarray] = []
        row_t: list[np.ndarray] = []
        for t in range(n_t):
            vol = np.asarray(arr[t])
            coords = detect_centroids(vol, detect_params)
            dets_by_t[t] = coords
            if coords.shape[0]:
                vol_norm = _normalize_intensity(
                    vol.astype(np.float32), detect_params.intensity_norm
                )
                feats, _ = extract_candidate_features(vol_norm, coords, detect_params)
                feat_rows.append(feats)
                row_t.append(np.full(coords.shape[0], t, dtype=np.int64))
        feats_all = (
            np.concatenate(feat_rows, axis=0)
            if feat_rows
            else np.zeros((0, 0))
        )
        row_t_all = (
            np.concatenate(row_t) if row_t else np.zeros(0, dtype=np.int64)
        )
        labels, weights = label_candidates(dets_by_t, gt, scale)
        cache[fam.name] = {
            "dets_by_t": dets_by_t,
            "feats": feats_all,
            "row_t": row_t_all,
            "labels": labels,
            "weights": weights,
            "gt": gt,
            "scale": scale,
            "n_true": n_true,
        }
        pos = int(labels.sum())
        print(
            f"[detect] {fam.name}: {time.time()-t0:.1f}s "
            f"dets={sum(len(v) for v in dets_by_t.values())} pos={pos} "
            f"({pos/max(len(labels),1)*100:.1f}%)",
            flush=True,
        )
    return cache


def cv_for_filter(cache, keep_by_fam, base_link):
    """Link each family's *filtered* detections and aggregate the CV."""
    rows = []
    for fam in CV_HOLDOUT:
        c = cache[fam.name]
        keep = keep_by_fam[fam.name]  # dict t -> bool mask
        filtered = {t: c["dets_by_t"][t][keep[t]] for t in c["dets_by_t"]}
        pred = link_centroids(filtered, scale=c["scale"], params=base_link)
        rows.append(score_family(fam, pred, c["gt"], c["n_true"], scale=c["scale"]))
    return aggregate(rows)


def keep_all(cache):
    return {
        fam.name: {
            t: np.ones(c.shape[0], dtype=bool)
            for t, c in cache[fam.name]["dets_by_t"].items()
        }
        for fam in CV_HOLDOUT
    }


def probs_to_keep(cache, fam_name, probs, thr):
    """Turn a flat per-candidate probability array into per-timepoint keep masks."""
    c = cache[fam_name]
    row_t = c["row_t"]
    flat_keep = probs >= thr
    keep: dict[int, np.ndarray] = {}
    for t in c["dets_by_t"]:
        sel = row_t == t
        keep[t] = flat_keep[sel]
    return keep


def edge_counts(res):
    return {
        r.name: {"tp": r.edge_tp, "fp": r.edge_fp, "fn": r.edge_fn,
                 "pred_nodes": r.num_pred_nodes}
        for r in res.per_dataset
    }


def summarise(res, baseline, base_ec, thr, kept_frac):
    per = {r.name: r.adj_edge_jaccard for r in res.per_dataset}
    no_reg = res.no_regression_vs(CHAMPION_PER_DATASET_ADJ)
    score_up = res.score > baseline.score + 1e-9
    ec = edge_counts(res)
    edge_delta = {
        name: {
            "d_tp": ec[name]["tp"] - base_ec[name]["tp"],
            "d_fp": ec[name]["fp"] - base_ec[name]["fp"],
            "d_fn": ec[name]["fn"] - base_ec[name]["fn"],
            "d_pred_nodes": ec[name]["pred_nodes"] - base_ec[name]["pred_nodes"],
        }
        for name in ec
    }
    tot = {k: sum(v[k] for v in edge_delta.values()) for k in ("d_tp", "d_fp", "d_fn")}
    rep = representativeness_report(res)
    return {
        "threshold": thr,
        "kept_fraction_by_family": kept_frac,
        "micro_adj_edge_jaccard": round(res.micro_adj_edge_jaccard, 4),
        "score": round(res.score, 4),
        "delta_score_vs_champion": round(res.score - baseline.score, 4),
        "no_per_dataset_regression": bool(no_reg),
        "promotable": bool(no_reg and score_up),
        "family_mix_sensitive": bool(rep.get("family_mix_sensitive")),
        "total_edge_delta": tot,
        "per_dataset_adj": {k: round(v, 4) for k, v in per.items()},
        "edge_delta_by_dataset": edge_delta,
    }


def main() -> int:
    cfg = load_champion_config()
    detect_params, base_link, _scale = champion_params(cfg)

    print("[stage] detect + features (once per family)", flush=True)
    cache = detect_and_features(detect_params)

    baseline = cv_for_filter(cache, keep_all(cache), base_link)
    base_ec = edge_counts(baseline)
    print(f"[baseline] champion micro_adj={baseline.micro_adj_edge_jaccard:.4f} "
          f"score={baseline.score:.4f}", flush=True)

    # LOFO: train one scorer per held-out family on the other three families.
    fam_names = [f.name for f in CV_HOLDOUT]
    probs_by_fam: dict[str, np.ndarray] = {}
    fold_info: dict[str, dict] = {}
    for held in fam_names:
        train_feats = np.concatenate(
            [cache[n]["feats"] for n in fam_names if n != held], axis=0
        )
        train_labels = np.concatenate(
            [cache[n]["labels"] for n in fam_names if n != held], axis=0
        )
        train_weights = np.concatenate(
            [cache[n]["weights"] for n in fam_names if n != held], axis=0
        )
        scorer = fit_scorer(train_feats, train_labels, train_weights)
        probs = scorer.probability(cache[held]["feats"])
        probs_by_fam[held] = probs
        # In-fold ranking diagnostic (train AUC-ish gap on held-out family).
        yl = cache[held]["labels"]
        gap = (
            float(probs[yl > 0.5].mean() - probs[yl < 0.5].mean())
            if (yl > 0.5).any() and (yl < 0.5).any() else float("nan")
        )
        fold_info[held] = {
            "coef": [round(float(v), 4) for v in scorer.coef],
            "intercept": round(float(scorer.intercept), 4),
            "heldout_pos_neg_prob_gap": round(gap, 4),
        }
        print(f"[fold] held={held} prob_gap(pos-neg)={gap:+.4f}", flush=True)

    variants = []
    for thr in THRESHOLDS:
        keep_by_fam = {}
        kept_frac = {}
        for fam in CV_HOLDOUT:
            keep_by_fam[fam.name] = probs_to_keep(cache, fam.name, probs_by_fam[fam.name], thr)
            total = sum(m.shape[0] for m in keep_by_fam[fam.name].values())
            kept = sum(int(m.sum()) for m in keep_by_fam[fam.name].values())
            kept_frac[fam.name] = round(kept / max(total, 1), 4)
        res = cv_for_filter(cache, keep_by_fam, base_link)
        row = summarise(res, baseline, base_ec, thr, kept_frac)
        variants.append(row)
        print(f"[thr] t={thr:.2f} micro_adj={res.micro_adj_edge_jaccard:.4f} "
              f"score={res.score:.4f} d={row['delta_score_vs_champion']:+.4f} "
              f"dTP={row['total_edge_delta']['d_tp']:+d} "
              f"dFP={row['total_edge_delta']['d_fp']:+d} "
              f"dFN={row['total_edge_delta']['d_fn']:+d} "
              f"no_reg={row['no_per_dataset_regression']} "
              f"promotable={row['promotable']}", flush=True)

    zero_row = next(v for v in variants if v["threshold"] == 0.0)
    sanity_reproduces_champion = (
        abs(zero_row["score"] - round(baseline.score, 4)) < 1e-9
        and zero_row["total_edge_delta"] == {"d_tp": 0, "d_fp": 0, "d_fn": 0}
    )

    variants_ranked = sorted(variants, key=lambda v: v["score"], reverse=True)
    promotable = [v for v in variants_ranked if v["promotable"]]

    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2828",
        "axis": "GT-learned portable detection scorer: pure-numpy logistic on a "
                "joint hand-crafted candidate feature vector (DoG response z-score, "
                "SOT-2829 appearance patch stats, Hessian blobness eigen-ratios, "
                "local density), leave-one-family-out training, keep-mask threshold "
                "sweep, same-seed A/B vs the frozen detect-link-dog-v4-shorttrack "
                "champion (detection+features cached once, only the keep-mask varies)",
        "cv_source": "biohub_tracking.eval.cv (SOT-2817 re-anchored full-metric "
                     "4-family leak-free holdout)",
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "evaluation_note": "leave-one-family-out: each family scored by a scorer "
                           "trained ONLY on the other three families (no per-family "
                           "GT leak). threshold=0.0 keeps all candidates => champion "
                           "byte-for-byte.",
        "baseline_champion": {
            "micro_adj_edge_jaccard": round(baseline.micro_adj_edge_jaccard, 4),
            "score": round(baseline.score, 4),
            "edge_counts_by_dataset": base_ec,
            "cv": cv_result_to_dict(baseline),
        },
        "champion_per_dataset_adj_floor": CHAMPION_PER_DATASET_ADJ,
        "lofo_folds": fold_info,
        "grid": {"threshold": THRESHOLDS},
        "threshold_zero_sanity_reproduces_champion": bool(sanity_reproduces_champion),
        "variants_ranked_by_score": variants_ranked,
        "n_promotable": len(promotable),
        "best_promotable": promotable[0] if promotable else None,
    }
    out = REPO / "experiments/sot2828/screen_detect_scorer.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"baseline champion score={baseline.score:.4f}")
    print(f"threshold=0 reproduces champion exactly: {sanity_reproduces_champion}")
    print(f"n_promotable={len(promotable)}")
    if not promotable:
        top = variants_ranked[0]
        print(f"NO PROMOTABLE variant. Top-by-score: t={top['threshold']} "
              f"score={top['score']} (delta {top['delta_score_vs_champion']:+}) "
              f"no_reg={top['no_per_dataset_regression']} "
              f"edge_delta={top['total_edge_delta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
