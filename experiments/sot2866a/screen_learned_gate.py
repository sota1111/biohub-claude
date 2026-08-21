"""Screen the learned edge-cost gate-EXPANSION FN-edge recovery (SOT-2870), leak-free.

SOT-2841 learned an edge re-ranker (``label = edge``) that was truly discriminative
(held-out ``p_edge`` gap 0.42) yet CV-neutral — because a *fixed raw-distance*
feasible set is already saturated (near ≈ GT), so re-ordering it recovers nothing.
SOT-2864 located the actual headroom: the feasibility **gate**. Its motion-model
linking with ``motion_gate_on_prediction=True`` scored +0.0111 with 4-family
non-regression purely by *admitting* raw-far but motion-consistent successors the
distance gate drops (**FN-edge recovery**).

This screens the union of those two: a learned edge classifier whose feature vector is
extended with the SOT-2864 **motion residual** (predicted-position distance) and two
Trackastra-style shallow **shape/intensity ratios** (brightness / spread change,
reused from the patch descriptor — no new extraction, no train/infer skew), used to
drive a **gate-EXPANSION admissibility**: a pair beyond ``max_distance`` in raw
distance is admitted only when it is motion-corrected in-range (``dist_pred <=
max_distance``), within a bounded raw ratio, **and** the classifier scores it a real
edge (``p_edge >= admit_prob``). Never an unbounded long-range edge.

**Leak-free (leave-one-family-out).** For each held-out family the model is trained on
the *other three* families' (edge-features, edge-labels) only — where a "feasible"
trainable pair now spans the **motion-corrected** union (raw ``<= max_distance`` OR
motion-predicted ``<= max_distance``), so the model actually sees the raw-far GT edges
it must learn to admit. Detection + descriptors are computed **once** per family; every
variant re-links off the cache (single-variable same-seed A/B).

Variants (all vs the frozen ``detect-link-dog-v4-shorttrack`` champion 0.6649):
  * ``champion``       — distance-only baseline (byte-frozen reference).
  * ``motion_rerank``  — SOT-2864 motion-model link, gate on RAW distance (re-rank
                          only, no learned model): isolates the motion re-rank.
  * ``motion_gate``    — SOT-2864 motion-model link, ``gate_on_prediction=True`` (the
                          +0.0111 pure motion gate, no learned filter): the axis to beat.
  * ``learned_gate@τ`` — THIS issue: motion-model link + learned gate-expansion at
                          admit-prob τ (the learned filter on the motion admit).

Writes ``experiments/sot2866a/screen_learned_gate.json``. No Kaggle submission.
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
from biohub_tracking.link import (
    LinkParams,
    _motion_field_predict,
    link_centroids,
)
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

MAX_DISTANCE = 7.0
# Motion-field prediction knobs — the SOT-2864 defaults (its promoted operating point).
MOTION_SMOOTH_SIGMA = 15.0
MOTION_GAIN = 1.0
# Gate-expansion knobs.
EXPAND_RATIO = 1.5              # raw dist cap = EXPAND_RATIO * MAX_DISTANCE
RERANK_WEIGHT = 1.0            # learned re-rank strength within the (expanded) set
ADMIT_PROBS = [0.5, 0.6, 0.7, 0.8, 0.9]  # gate-expansion admit-prob sweep


def _base_link_params(**kw) -> LinkParams:
    """A LinkParams sharing the champion's scored knobs (mtl=4, no in-linker division)."""
    return LinkParams(
        max_distance=MAX_DISTANCE,
        allow_division=False,
        division_distance=MAX_DISTANCE,
        min_track_length=4,
        **kw,
    )


def _motion_pred_dist(src, dst, scale_arr):
    """``(P, Q)`` scaled distance from each motion-predicted src to each dst.

    Uses the exact ``_motion_field_predict`` the linker calls (same anchors, sigma,
    gain), so the training feature ``motion_resid`` equals the inference one.
    """
    src_pred = _motion_field_predict(
        src, dst, scale_arr, MAX_DISTANCE, MOTION_SMOOTH_SIGMA, MOTION_GAIN
    )
    dpred = (src_pred[:, None, :] - dst[None, :, :]) * scale_arr
    return np.sqrt((dpred**2).sum(axis=2))


def build_family_training(dets, descs, gt, scale):
    """Edge-features + 0/1 labels over the **motion-corrected** feasible union.

    A trainable pair ``(i @ t) -> (j @ t+1)`` is feasible iff its raw scaled distance
    OR its motion-predicted distance is ``<= MAX_DISTANCE`` (the union the gate
    expansion admits from). Labelling mirrors SOT-2841 (matched-source positives are
    GT consecutive edges; matched-source non-GT successors are confident negatives;
    unmatched sources are skipped to avoid node-PU contamination) — but now the
    raw-far, motion-corrected GT edges (the FN edges) are IN the trainable set.
    """
    scale_arr = np.asarray(scale, dtype=float)
    pred = link_centroids(dets, scale=scale, params=_base_link_params())
    p2g = match_nodes(pred, gt, scale=tuple(scale_arr), max_distance=MAX_DISTANCE)

    ids_by_t: dict[int, list[int]] = {}
    nid = 0
    for t in sorted(dets):
        ids_by_t[t] = list(range(nid, nid + len(dets[t])))
        nid += len(dets[t])

    gt_edges = set(gt.edges)
    feats: list[np.ndarray] = []
    labels: list[float] = []
    n_expanded_pos = 0
    times = sorted(dets)
    for t_a, t_b in zip(times, times[1:]):
        if t_b != t_a + 1:
            continue
        src, dst = dets[t_a], dets[t_b]
        if len(src) == 0 or len(dst) == 0:
            continue
        diff = (src[:, None, :] - dst[None, :, :]) * scale_arr
        dist = np.sqrt((diff**2).sum(axis=2))
        dist_pred = _motion_pred_dist(src, dst, scale_arr)
        planes = edge_feature_planes(
            dist, descs[t_a], descs[t_b], MAX_DISTANCE, dist_pred=dist_pred
        )
        feasible = (dist <= MAX_DISTANCE) | (dist_pred <= MAX_DISTANCE)
        src_ids, dst_ids = ids_by_t[t_a], ids_by_t[t_b]
        fi, fj = np.where(feasible)
        for i, j in zip(fi.tolist(), fj.tolist()):
            gu = p2g.get(src_ids[i])
            if gu is None:
                continue  # unmatched source: PU-ambiguous, skip
            gv = p2g.get(dst_ids[j])
            is_pos = gv is not None and (gu, gv) in gt_edges
            feats.append(planes[i, j])
            labels.append(1.0 if is_pos else 0.0)
            if is_pos and dist[i, j] > MAX_DISTANCE:
                n_expanded_pos += 1  # a raw-far GT edge only the expansion can admit
    if not feats:
        return np.zeros((0, len(EDGE_FEATURE_NAMES))), np.zeros(0), 0
    return (
        np.asarray(feats, dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
        n_expanded_pos,
    )


def _cv_over_families(cache, link_of_fam):
    """Aggregate a per-family LinkParams factory over the leak-free 4-family CV."""
    rows = []
    for fam in CV_HOLDOUT:
        dets, descs, gt, scale, n_true = cache[fam.name]
        params, use_desc = link_of_fam(fam)
        pred = link_centroids(
            dets, scale=scale, params=params,
            descriptors=descs if use_desc else None,
        )
        rows.append(score_family(fam, pred, gt, n_true, scale=scale))
    return aggregate(rows)


def edge_counts_by_dataset(res) -> dict:
    return {
        r.name: {"tp": r.edge_tp, "fp": r.edge_fp, "fn": r.edge_fn,
                 "pred_nodes": r.num_pred_nodes}
        for r in res.per_dataset
    }


def summarise(name, res, baseline, base_ec, extra=None):
    per = {r.name: r.adj_edge_jaccard for r in res.per_dataset}
    no_reg = res.no_regression_vs(CHAMPION_PER_DATASET_ADJ)
    score_up = res.score > baseline.score + 1e-9
    ec = edge_counts_by_dataset(res)
    edge_delta = {
        nm: {
            "d_tp": ec[nm]["tp"] - base_ec[nm]["tp"],
            "d_fp": ec[nm]["fp"] - base_ec[nm]["fp"],
            "d_fn": ec[nm]["fn"] - base_ec[nm]["fn"],
            "d_pred_nodes": ec[nm]["pred_nodes"] - base_ec[nm]["pred_nodes"],
        }
        for nm in ec
    }
    tot = {
        "d_tp": sum(v["d_tp"] for v in edge_delta.values()),
        "d_fp": sum(v["d_fp"] for v in edge_delta.values()),
        "d_fn": sum(v["d_fn"] for v in edge_delta.values()),
    }
    rep = representativeness_report(res)
    row = {
        "variant": name,
        "micro_adj_edge_jaccard": round(res.micro_adj_edge_jaccard, 4),
        "score": round(res.score, 4),
        "delta_score_vs_champion": round(res.score - baseline.score, 4),
        "no_per_dataset_regression": bool(no_reg),
        "promotable": bool(no_reg and score_up),
        "family_mix_sensitive": bool(rep.get("family_mix_sensitive")),
        "micro_lineage_macro_gap": rep.get("micro_lineage_macro_gap"),
        "total_edge_delta": tot,
        "per_dataset_adj": {k: round(v, 4) for k, v in per.items()},
        "edge_delta_by_dataset": edge_delta,
    }
    if extra:
        row.update(extra)
    return row


def main() -> int:
    cfg = load_champion_config()
    detect, base_link, _cfg_scale = champion_params(cfg)

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
        X, y, n_exp = build_family_training(dets, descs, gt, scale)
        training[fam.name] = (X, y)
        print(f"[detect] {fam.name}: {time.time()-t0:.1f}s "
              f"dets={sum(len(v) for v in dets.values())} "
              f"train_pairs={len(y)} pos={int(y.sum())} "
              f"expanded_pos(raw>max)={n_exp}", flush=True)

    # --- Baseline A0: distance-only champion (byte-frozen reference). ---
    baseline = _cv_over_families(cache, lambda fam: (base_link, False))
    base_ec = edge_counts_by_dataset(baseline)
    print(f"[champion] micro_adj={baseline.micro_adj_edge_jaccard:.4f} "
          f"score={baseline.score:.4f}", flush=True)

    variants = []
    zero_sanity = summarise("champion", baseline, baseline, base_ec)
    variants.append(zero_sanity)

    # --- A1: SOT-2864 motion re-rank (gate on raw distance, no learned model). ---
    res = _cv_over_families(
        cache,
        lambda fam: (_base_link_params(motion_model_link=True,
                                       motion_gate_on_prediction=False), False),
    )
    variants.append(summarise("motion_rerank", res, baseline, base_ec))
    print(f"[motion_rerank] score={res.score:.4f} "
          f"d={res.score - baseline.score:+.4f}", flush=True)

    # --- A2: SOT-2864 pure motion GATE (gate_on_prediction, no learned filter). ---
    res = _cv_over_families(
        cache,
        lambda fam: (_base_link_params(motion_model_link=True,
                                       motion_gate_on_prediction=True), False),
    )
    variants.append(summarise("motion_gate", res, baseline, base_ec))
    print(f"[motion_gate] score={res.score:.4f} "
          f"d={res.score - baseline.score:+.4f} "
          f"no_reg={variants[-1]['no_per_dataset_regression']}", flush=True)

    # --- B: learned gate-EXPANSION at each admit-prob. Model trained leak-free. ---
    # Pre-fit each held-out family's LOFO model once (weight-invariant across τ).
    lofo_model = {}
    gaps = {}
    for fam in CV_HOLDOUT:
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
        model = fit_edge_cost(X, y, weight=RERANK_WEIGHT)
        lofo_model[fam.name] = model
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
    mean_gap = (round(sum(g["gap"] for g in gaps.values()) / len(gaps), 4)
                if gaps else None)
    print(f"[lofo] mean held-out p_edge gap={mean_gap}", flush=True)

    for tau in ADMIT_PROBS:
        def link_of_fam(fam, tau=tau):
            return (
                _base_link_params(
                    motion_model_link=True,
                    motion_gate_on_prediction=False,   # base gate on RAW; expand admits
                    edge_cost_model=lofo_model[fam.name].to_dict(),
                    edge_gate_expand=True,
                    edge_gate_admit_prob=tau,
                    edge_gate_expand_ratio=EXPAND_RATIO,
                ),
                True,
            )
        res = _cv_over_families(cache, link_of_fam)
        row = summarise(f"learned_gate@{tau}", res, baseline, base_ec,
                        extra={"admit_prob": tau})
        variants.append(row)
        print(f"[learned_gate@{tau}] score={res.score:.4f} "
              f"d={row['delta_score_vs_champion']:+.4f} "
              f"dTP={row['total_edge_delta']['d_tp']:+d} "
              f"dFP={row['total_edge_delta']['d_fp']:+d} "
              f"dFN={row['total_edge_delta']['d_fn']:+d} "
              f"no_reg={row['no_per_dataset_regression']} "
              f"promotable={row['promotable']}", flush=True)

    champ_row = next(v for v in variants if v["variant"] == "champion")
    sanity_reproduces_champion = (
        abs(champ_row["score"] - round(baseline.score, 4)) < 1e-9
        and champ_row["total_edge_delta"] == {"d_tp": 0, "d_fp": 0, "d_fn": 0}
    )
    ranked = sorted(variants, key=lambda v: v["score"], reverse=True)
    promotable = [v for v in ranked
                  if v["promotable"] and v["variant"].startswith("learned_gate")]

    payload = {
        "recordedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issue": "SOT-2870",
        "axis": "learned edge-cost gate-EXPANSION (motion residual + shape/intensity) "
                "FN-edge recovery: leave-one-family-out logistic over the SOT-2841 "
                "edge features PLUS SOT-2864 motion residual + descriptor "
                "intensity/spread ratios; a raw-far pair is admitted iff "
                "motion-corrected in-range AND raw<=ratio*max AND p_edge>=admit_prob "
                "(never an unbounded long-range edge). Same-seed A/B vs the frozen "
                "detect-link-dog-v4-shorttrack champion and vs the SOT-2864 motion gate.",
        "cv_source": "biohub_tracking.eval.cv (SOT-2817 re-anchored full-metric "
                     "4-family leak-free holdout)",
        "leak_free_protocol": "leave-one-family-out over the motion-corrected feasible "
                              "union (raw<=max OR motion-pred<=max); positives=GT "
                              "consecutive edges, negatives=feasible non-GT successors "
                              "of matched sources, unmatched-source pairs excluded.",
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "feature_names": list(EDGE_FEATURE_NAMES),
        "knobs": {
            "max_distance": MAX_DISTANCE,
            "motion_smooth_sigma": MOTION_SMOOTH_SIGMA,
            "motion_gain": MOTION_GAIN,
            "expand_ratio": EXPAND_RATIO,
            "rerank_weight": RERANK_WEIGHT,
            "admit_probs": ADMIT_PROBS,
        },
        "baseline_champion": {
            "micro_adj_edge_jaccard": round(baseline.micro_adj_edge_jaccard, 4),
            "score": round(baseline.score, 4),
            "edge_counts_by_dataset": base_ec,
            "cv": cv_result_to_dict(baseline),
        },
        "champion_per_dataset_adj_floor": CHAMPION_PER_DATASET_ADJ,
        "champion_sanity_reproduced": bool(sanity_reproduces_champion),
        "mean_held_out_prob_gap": mean_gap,
        "held_out_prob_gap": gaps,
        "variants_ranked_by_score": ranked,
        "n_promotable_learned_gate": len(promotable),
        "best_promotable": promotable[0] if promotable else None,
    }
    out = REPO / "experiments/sot2866a/screen_learned_gate.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(f"champion score={baseline.score:.4f} sanity_reproduced={sanity_reproduces_champion}")
    print(f"mean held-out p_edge gap={mean_gap}")
    if promotable:
        b = promotable[0]
        print(f"BEST PROMOTABLE learned_gate: {b['variant']} score={b['score']} "
              f"(+{b['delta_score_vs_champion']}) edge_delta={b['total_edge_delta']}")
    else:
        top = next(v for v in ranked if v["variant"].startswith("learned_gate"))
        print(f"NO PROMOTABLE learned_gate. Best learned_gate by score: {top['variant']} "
              f"score={top['score']} (delta {top['delta_score_vs_champion']:+}) "
              f"no_reg={top['no_per_dataset_regression']} "
              f"edge_delta={top['total_edge_delta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
