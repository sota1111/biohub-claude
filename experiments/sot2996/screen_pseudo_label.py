"""Screen semi-supervised dense pseudo-label self-training for the detection
scorer (SOT-2996).

Thesis. The learned scorer (SOT-2828) failed because sparse GT poisons its labels:
every un-annotated real cell is trained as a (down-weighted) negative, so the scorer
ranks real cells *below* noise (held-out pos−neg prob gap NEGATIVE on 3/4 families).
This screen densifies the teacher signal — the champion detector's own
high-confidence, temporally-consistent detections become **dense pseudo-positives**
(:mod:`biohub_tracking.pseudo_label`) — and asks whether a scorer trained on the
densified labels (B) beats one trained on the SOT-2828 sparse-GT PU labels (A), both
same-seed A/B vs the byte-frozen ``detect-link-dog-v4-shorttrack`` champion.

Leak-free = leave-one-family-out (LOFO), identical to SOT-2828: each of the four CV
families is scored by a scorer trained ONLY on the other three; per-family held-out
predictions aggregate through the SOT-2817 re-anchored full-metric CV
(:mod:`biohub_tracking.eval.cv`), byte-comparable to the registry champion. The
pseudo-label confidence gate threshold is computed on the **pooled training
families** per fold, never on the held-out family (leak-free). ``threshold=0.0``
keeps every candidate ⇒ reproduces the champion byte-for-byte (sanity). Detection +
features + champion track-length are cached once; only the labeling scheme and the
keep-mask change between arms. No Kaggle submission.

Writes ``experiments/sot2996/screen_pseudo_label.json``.
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
    FEATURE_NAMES,
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
from biohub_tracking.pseudo_label import (
    PseudoLabelConfig,
    densify_labels,
    track_length_per_candidate,
)

REPO = Path(__file__).resolve().parents[2]

# Frozen champion per-dataset adjusted-edge-Jaccard floor (non-regression gate),
# from the registry champion CV (identical to SOT-2828's floor).
CHAMPION_PER_DATASET_ADJ = {
    "44b6_0113de3b": 0.8895,
    "44b6_0b24845f": 0.6817,
    "6bba_05b6850b": 0.5700,
    "6bba_05db0fb1": 0.7310,
}

# 6bba density regimes (SOT-2921 observable covariate): the sparse-tail sequence vs
# the dense one — the pair the density-mix wall crosscuts. Recorded per-arm so the
# node-count-penalty effect on *both* 6bba regimes is explicit.
REGIME = {
    "6bba_05b6850b": "sparse",
    "6bba_05db0fb1": "dense",
    "44b6_0113de3b": "dense",
    "44b6_0b24845f": "dense",
}

THRESHOLDS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]
RESP_PERCENTILE = 90.0
MIN_TRACK_LEN = 3
PSEUDO_WEIGHT = 0.5


def detect_and_features(detect_params, base_link):
    """Per family: detect per timepoint, extract features + sparse-GT labels +
    champion track length (temporal-consistency signal, cached once)."""
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
        feats_all = np.concatenate(feat_rows, axis=0) if feat_rows else np.zeros((0, 0))
        row_t_all = np.concatenate(row_t) if row_t else np.zeros(0, dtype=np.int64)
        # SOT-2828 sparse-GT PU labels (arm A) + champion track length (for arm B).
        labels_a, weights_a = label_candidates(dets_by_t, gt, scale)
        track_len = track_length_per_candidate(dets_by_t, scale, base_link)
        resp_z = (
            feats_all[:, FEATURE_NAMES.index("resp_z")]
            if feats_all.shape[0]
            else np.zeros(0)
        )
        cache[fam.name] = {
            "dets_by_t": dets_by_t,
            "feats": feats_all,
            "row_t": row_t_all,
            "resp_z": np.asarray(resp_z, dtype=np.float64),
            "track_len": track_len,
            "labels_a": labels_a,
            "weights_a": weights_a,
            "gt": gt,
            "scale": scale,
            "n_true": n_true,
        }
        pos = int(labels_a.sum())
        print(
            f"[detect] {fam.name}: {time.time()-t0:.1f}s "
            f"dets={sum(len(v) for v in dets_by_t.values())} gt_pos={pos} "
            f"({pos/max(len(labels_a),1)*100:.1f}%) "
            f"mean_tracklen={track_len.mean():.2f}",
            flush=True,
        )
    return cache


def build_arm_b_labels(cache, fam_names):
    """Dense pseudo-label every family (arm B), with the confidence-gate threshold
    computed per LOFO fold on the pooled TRAINING families (leak-free)."""
    cfg = PseudoLabelConfig(
        resp_percentile=RESP_PERCENTILE,
        min_track_len=MIN_TRACK_LEN,
        pseudo_weight=PSEUDO_WEIGHT,
    )
    # Per-fold pooled-train resp_z percentile threshold (leak-free: excludes held).
    fold_threshold: dict[str, float] = {}
    for held in fam_names:
        pool = np.concatenate(
            [cache[n]["resp_z"] for n in fam_names if n != held]
        )
        fold_threshold[held] = float(np.percentile(pool, RESP_PERCENTILE))
    return cfg, fold_threshold


def cv_for_filter(cache, keep_by_fam, base_link):
    rows = []
    for fam in CV_HOLDOUT:
        c = cache[fam.name]
        keep = keep_by_fam[fam.name]
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
    c = cache[fam_name]
    row_t = c["row_t"]
    flat_keep = probs >= thr
    keep: dict[int, np.ndarray] = {}
    for t in c["dets_by_t"]:
        keep[t] = flat_keep[row_t == t]
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
    tot = {k: sum(v[k] for v in edge_delta.values())
           for k in ("d_tp", "d_fp", "d_fn", "d_pred_nodes")}
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
        "regime_6bba": {
            "sparse_6bba_05b6850b": round(per.get("6bba_05b6850b", float("nan")), 4),
            "dense_6bba_05db0fb1": round(per.get("6bba_05db0fb1", float("nan")), 4),
        },
        "edge_delta_by_dataset": edge_delta,
    }


def run_arm(name, cache, base_link, baseline, base_ec, label_fn, fam_names):
    """One labeling arm: LOFO-train, held-out prob-gap diagnostic, threshold sweep."""
    probs_by_fam: dict[str, np.ndarray] = {}
    fold_info: dict[str, dict] = {}
    for held in fam_names:
        tr_feats, tr_labels, tr_weights = [], [], []
        for nfam in fam_names:
            if nfam == held:
                continue
            lab, wt = label_fn(nfam, held)
            tr_feats.append(cache[nfam]["feats"])
            tr_labels.append(lab)
            tr_weights.append(wt)
        scorer = fit_scorer(
            np.concatenate(tr_feats, axis=0),
            np.concatenate(tr_labels, axis=0),
            np.concatenate(tr_weights, axis=0),
        )
        probs = scorer.probability(cache[held]["feats"])
        probs_by_fam[held] = probs
        # prob-gap is always measured against the SOT-2828 SPARSE-GT labels: does the
        # scorer rank real (GT-annotated) cells above noise? That is the PU-repair KPI.
        yl = cache[held]["labels_a"]
        gap = (
            float(probs[yl > 0.5].mean() - probs[yl < 0.5].mean())
            if (yl > 0.5).any() and (yl < 0.5).any() else float("nan")
        )
        fold_info[held] = {
            "coef": [round(float(v), 4) for v in scorer.coef],
            "intercept": round(float(scorer.intercept), 4),
            "heldout_pos_neg_prob_gap": round(gap, 4),
        }
        print(f"  [{name} fold] held={held} prob_gap(pos-neg)={gap:+.4f}", flush=True)

    variants = []
    for thr in THRESHOLDS:
        keep_by_fam, kept_frac = {}, {}
        for fam in CV_HOLDOUT:
            keep_by_fam[fam.name] = probs_to_keep(
                cache, fam.name, probs_by_fam[fam.name], thr
            )
            total = sum(m.shape[0] for m in keep_by_fam[fam.name].values())
            kept = sum(int(m.sum()) for m in keep_by_fam[fam.name].values())
            kept_frac[fam.name] = round(kept / max(total, 1), 4)
        res = cv_for_filter(cache, keep_by_fam, base_link)
        row = summarise(res, baseline, base_ec, thr, kept_frac)
        variants.append(row)
        print(f"  [{name} thr] t={thr:.2f} micro_adj={res.micro_adj_edge_jaccard:.4f} "
              f"d={row['delta_score_vs_champion']:+.4f} "
              f"dTP={row['total_edge_delta']['d_tp']:+d} "
              f"dFP={row['total_edge_delta']['d_fp']:+d} "
              f"dNodes={row['total_edge_delta']['d_pred_nodes']:+d} "
              f"no_reg={row['no_per_dataset_regression']} "
              f"promotable={row['promotable']}", flush=True)
    variants_ranked = sorted(variants, key=lambda v: v["score"], reverse=True)
    promotable = [v for v in variants_ranked if v["promotable"]]
    mean_gap = float(
        np.nanmean([fold_info[h]["heldout_pos_neg_prob_gap"] for h in fam_names])
    )
    return {
        "lofo_folds": fold_info,
        "mean_heldout_prob_gap": round(mean_gap, 4),
        "variants_ranked_by_score": variants_ranked,
        "n_promotable": len(promotable),
        "best_promotable": promotable[0] if promotable else None,
    }


def main() -> int:
    cfg_champ = load_champion_config()
    detect_params, base_link, _scale = champion_params(cfg_champ)

    print("[stage] detect + features + track-length (once per family)", flush=True)
    cache = detect_and_features(detect_params, base_link)

    baseline = cv_for_filter(cache, keep_all(cache), base_link)
    base_ec = edge_counts(baseline)
    print(f"[baseline] champion micro_adj={baseline.micro_adj_edge_jaccard:.4f} "
          f"score={baseline.score:.4f}", flush=True)

    fam_names = [f.name for f in CV_HOLDOUT]

    # ---- Arm A: SOT-2828 sparse-GT PU labels -------------------------------
    def label_a(fam_name, _held):
        return cache[fam_name]["labels_a"], cache[fam_name]["weights_a"]

    print("[arm A] sparse-GT PU labels (SOT-2828 baseline labeling)", flush=True)
    arm_a = run_arm("A", cache, base_link, baseline, base_ec, label_a, fam_names)

    # ---- Arm B: dense pseudo-labels ---------------------------------------
    pl_cfg, fold_threshold = build_arm_b_labels(cache, fam_names)
    pseudo_diag: dict[str, dict] = {}

    def label_b(fam_name, held):
        c = cache[fam_name]
        labels, weights, diag = densify_labels(
            c["dets_by_t"], c["feats"], FEATURE_NAMES, c["gt"], c["scale"],
            base_link, pl_cfg,
            resp_z_threshold=fold_threshold[held],
            track_len=c["track_len"],
        )
        # Record diagnostics once per family (fold thresholds are close; keep the
        # first held-fold's densification snapshot for the report).
        pseudo_diag.setdefault(fam_name, diag.to_dict())
        return labels, weights

    print("[arm B] dense pseudo-labels (confidence+consistency gate)", flush=True)
    arm_b = run_arm("B", cache, base_link, baseline, base_ec, label_b, fam_names)

    zero_a = next(v for v in arm_a["variants_ranked_by_score"] if v["threshold"] == 0.0)
    zero_b = next(v for v in arm_b["variants_ranked_by_score"] if v["threshold"] == 0.0)
    sanity = (
        abs(zero_a["score"] - round(baseline.score, 4)) < 1e-9
        and zero_a["total_edge_delta"]["d_tp"] == 0
        and zero_a["total_edge_delta"]["d_fp"] == 0
        and abs(zero_b["score"] - round(baseline.score, 4)) < 1e-9
    )

    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2996",
        "axis": "semi-supervised dense pseudo-label self-training: champion "
                "high-confidence (resp_z >= p90) + temporally-consistent "
                "(champion track length >= 3) un-annotated detections promoted to "
                "dense pseudo-positives to repair the SOT-2828 sparse-GT PU labeling; "
                "LOFO scorer A/B (A=sparse-GT PU labels, B=dense pseudo-labels), "
                "keep-mask threshold sweep, same-seed vs frozen champion.",
        "cv_source": "biohub_tracking.eval.cv (SOT-2817 re-anchored full-metric "
                     "4-family leak-free holdout)",
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "leak_free_note": "LOFO: each family scored by a scorer trained ONLY on the "
                          "other three. Pseudo-label confidence-gate threshold "
                          "computed per fold on pooled TRAINING families (held-out "
                          "excluded). Consistency gate = GT-free champion track "
                          "length. threshold=0.0 keeps all => champion byte-for-byte.",
        "gate_config": {
            "resp_percentile": RESP_PERCENTILE,
            "min_track_len": MIN_TRACK_LEN,
            "pseudo_weight": PSEUDO_WEIGHT,
            "fold_resp_z_threshold": {k: round(v, 4) for k, v in fold_threshold.items()},
        },
        "regime_6bba": REGIME,
        "baseline_champion": {
            "micro_adj_edge_jaccard": round(baseline.micro_adj_edge_jaccard, 4),
            "score": round(baseline.score, 4),
            "edge_counts_by_dataset": base_ec,
            "cv": cv_result_to_dict(baseline),
        },
        "champion_per_dataset_adj_floor": CHAMPION_PER_DATASET_ADJ,
        "threshold_zero_sanity_reproduces_champion": bool(sanity),
        "pseudo_label_diagnostics_by_family": pseudo_diag,
        "arm_A_sparse_gt_pu": arm_a,
        "arm_B_dense_pseudo": arm_b,
        "prob_gap_repair": {
            "arm_A_mean_heldout_prob_gap": arm_a["mean_heldout_prob_gap"],
            "arm_B_mean_heldout_prob_gap": arm_b["mean_heldout_prob_gap"],
            "improved": bool(
                arm_b["mean_heldout_prob_gap"] > arm_a["mean_heldout_prob_gap"]
            ),
        },
        "any_promotable": bool(arm_a["n_promotable"] or arm_b["n_promotable"]),
    }
    out = REPO / "experiments/sot2996/screen_pseudo_label.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nwrote {out}")
    print(f"threshold=0 reproduces champion (both arms): {sanity}")
    print(f"prob_gap A={arm_a['mean_heldout_prob_gap']:+.4f} "
          f"B={arm_b['mean_heldout_prob_gap']:+.4f} "
          f"repaired={payload['prob_gap_repair']['improved']}")
    print(f"n_promotable A={arm_a['n_promotable']} B={arm_b['n_promotable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
