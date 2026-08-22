"""SOT-2994 A/B: learned CROSS-ATTENTION edge linking (SimpleNodeTransformer port).

Leak-free leave-one-family-out (LOFO) same-seed A/B of a learned cross-attention
edge scorer vs the frozen ARGUS motion-model champion (config_sha256 f2b1076…6522fc,
CV micro_adj 0.6760).

The scorer (``biohub_tracking.xattn_edge.CrossAttentionEdgeCost``) contextualises each
edge embedding by attention over its source's competing successors and its
destination's competing predecessors before scoring ``p_edge`` — the official
``SimpleNodeTransformer`` idea, the structural differentiator from the REJECTED
per-edge logistic SOT-2841 (held-out gap 0.42 → zero CV gain) / SOT-2870 gate. The
learned ``p_edge`` enters the champion cost as ``dist + weight·(1 − p_edge)``
(re-rank only; the ``≤ max_distance`` motion gate is unchanged → metric-valid).

Protocol (mirrors experiments/sot2922 / sot2841):
* **Detection frozen** at champion params + descriptors, cached ONCE per family
  (linking-only ablation; SOT-2993 learned-detector node features are used if a config
  supplies them, else the handcrafted geometric+intensity edge features here — so this
  axis is evaluable standalone without SOT-2993).
* **Masked-sparse supervision** (the official baseline's masked loss): a trainable pair
  is a champion-feasible pair whose SOURCE is GT-matched — positive iff it is the GT
  edge, negative otherwise (its other feasible successors); unmatched-source pairs are
  excluded, so there is no SOT-2828 node positive-unlabeled contamination.
* **Leak-free LOFO**: each held-out family is linked by a model trained ONLY on the
  other three families' transitions; the four held-out predictions aggregate into the
  leak-free CV. Model coefficients are weight-independent, so one fit per fold is swept
  over the re-rank strengths.

Promotion gate (two-signal, same as every biohub child): conditional CV must clear
**4/4 per-dataset non-regression** (adjusted AND raw) AND beat champion micro_adj
0.6760. Primary = micro_adj (royerlab adjusted edge Jaccard); guardrail = micro_raw.
Champion config stays byte-frozen; NO Kaggle submission.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from biohub_tracking.champion import champion_params, load_champion_config
from biohub_tracking.detect import detect_volume_series_with_descriptors
from biohub_tracking.edge_linker import EDGE_FEATURE_NAMES, edge_feature_planes
from biohub_tracking.eval.cv import (
    CV_HOLDOUT,
    aggregate,
    cv_result_to_dict,
    representativeness_report,
    score_family,
)
from biohub_tracking.io import geff_estimated_num_nodes, geff_scale, load_geff
from biohub_tracking.link import LinkParams, _motion_field_predict, link_centroids
from biohub_tracking.matching import match_nodes
from biohub_tracking.pipeline import _open_image_array
from biohub_tracking.xattn_edge import fit_cross_attention, torch_available

REPO = Path(__file__).resolve().parents[2]
CHAMPION_CONFIG = REPO / "champion/config.json"
CHAMPION_CONFIG_SHA256 = (
    "f2b107674d870cfd8e1b667a5d487b15b994382f9de0e9c3bc66a0c05b6522fc"
)
CHAMPION_REFERENCE_MICRO_ADJ = 0.6760
OUT = REPO / "experiments/sot2994/ab_xattn_edge.json"

MAX_DISTANCE = 7.0
WEIGHTS = [0.0, 0.5, 1.0, 2.0, 4.0]
EPOCHS = 60
SEED = 0
MODEL_D = 8  # embed/context width (small: ~360-param scorer)
# Training-transition subsample stride: the dense 6bba frames (~750 dets) make the
# per-epoch autograd O(minutes); a stride-2 subsample of frame pairs cuts training
# compute ~2× while keeping >800 positive GT edges/fold. The full CV is ALWAYS
# scored on EVERY frame (inference), so this only bounds the training budget.
TRAIN_STRIDE = 2
CACHE = REPO / "experiments/sot2994/cache"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _transitions_for_family(dets, descs, gt, scale, link):
    """Per-transition training/inference tensors for one family (champion motion gate).

    Each transition dict carries ``planes`` (P,Q,F edge features WITH the champion
    motion-predicted residual), ``feasible`` (the union candidate set the attention
    pools over, matching inference), ``trainable`` (feasible & GT-matched source), and
    0/1 ``labels`` (GT edge). Also returns the flat held-out feature rows + labels for
    the discriminative-gap diagnostic.
    """
    scale_arr = np.asarray(scale, dtype=float)
    # A distance-only graph only to get the pred node ids per timepoint for matching.
    pred = link_centroids(dets, scale=scale, params=LinkParams(max_distance=MAX_DISTANCE))
    p2g = match_nodes(pred, gt, scale=tuple(scale_arr), max_distance=MAX_DISTANCE)
    ids_by_t: dict[int, list[int]] = {}
    nid = 0
    for t in sorted(dets):
        ids_by_t[t] = list(range(nid, nid + len(dets[t])))
        nid += len(dets[t])
    gt_edges = set(gt.edges)

    transitions = []
    flat_feats: list[np.ndarray] = []
    flat_labels: list[float] = []
    times = sorted(dets)
    for t_a, t_b in zip(times, times[1:]):
        if t_b != t_a + 1:
            continue
        src, dst = dets[t_a], dets[t_b]
        if len(src) == 0 or len(dst) == 0:
            continue
        diff = (src[:, None, :] - dst[None, :, :]) * scale_arr
        dist = np.sqrt((diff**2).sum(axis=2))
        # Champion motion-predicted residual (SOT-2864), same knobs the linker uses.
        src_pred = _motion_field_predict(
            src, dst, scale_arr, link.max_distance,
            link.motion_smooth_sigma, link.motion_gain,
        )
        diff_pred = (src_pred[:, None, :] - dst[None, :, :]) * scale_arr
        dist_pred = np.sqrt((diff_pred**2).sum(axis=2))
        planes = edge_feature_planes(
            dist, descs[t_a], descs[t_b], MAX_DISTANCE, dist_pred=dist_pred
        )
        feasible = (dist <= MAX_DISTANCE) | (dist_pred <= MAX_DISTANCE)
        src_ids, dst_ids = ids_by_t[t_a], ids_by_t[t_b]
        labels = np.zeros(dist.shape, dtype=float)
        trainable = np.zeros(dist.shape, dtype=bool)
        fi, fj = np.where(feasible)
        for i, j in zip(fi.tolist(), fj.tolist()):
            gu = p2g.get(src_ids[i])
            if gu is None:
                continue  # unmatched source: PU-ambiguous, skip (masked loss)
            gv = p2g.get(dst_ids[j])
            is_pos = gv is not None and (gu, gv) in gt_edges
            trainable[i, j] = True
            labels[i, j] = 1.0 if is_pos else 0.0
            flat_feats.append(planes[i, j])
            flat_labels.append(1.0 if is_pos else 0.0)
        transitions.append(
            {"planes": planes, "feasible": feasible,
             "trainable": trainable, "labels": labels}
        )
    Xf = (np.asarray(flat_feats, dtype=float)
          if flat_feats else np.zeros((0, len(EDGE_FEATURE_NAMES))))
    yf = np.asarray(flat_labels, dtype=float) if flat_labels else np.zeros(0)
    return transitions, Xf, yf


def main() -> int:
    if not torch_available():
        raise SystemExit("torch unavailable — training the cross-attention scorer needs it")

    champ_cfg = load_champion_config(CHAMPION_CONFIG)
    detect_params, champ_link, _scale = champion_params(champ_cfg)

    gt_by, scale_by, ntrue_by = {}, {}, {}
    for fam in CV_HOLDOUT:
        geff = REPO / fam.geff
        gt_by[fam.name] = load_geff(geff)
        scale_by[fam.name] = geff_scale(geff)
        ntrue_by[fam.name] = geff_estimated_num_nodes(geff)

    # ---- Detect ONCE per family (frozen champion + descriptors) --------------
    dets_by, descs_by = {}, {}
    trans_by, flatX_by, flaty_by = {}, {}, {}
    CACHE.mkdir(parents=True, exist_ok=True)
    for fam in CV_HOLDOUT:
        t0 = time.time()
        cache_f = CACHE / f"dets_{fam.name}.npz"
        if cache_f.exists():
            z = np.load(cache_f, allow_pickle=True)
            n = int(z["n_t"])
            dets = {int(t): z[f"d_{t}"] for t in range(n)}
            descs = {int(t): z[f"s_{t}"] for t in range(n)}
        else:
            arr = _open_image_array(REPO / fam.image)
            dets, descs = detect_volume_series_with_descriptors(
                arr, detect_params, scale=scale_by[fam.name]
            )
            save = {"n_t": len(dets)}
            for t in dets:
                save[f"d_{t}"] = dets[t]
                save[f"s_{t}"] = descs[t]
            np.savez_compressed(cache_f, **save)
        dets_by[fam.name], descs_by[fam.name] = dets, descs
        tr, Xf, yf = _transitions_for_family(
            dets, descs, gt_by[fam.name], scale_by[fam.name], champ_link
        )
        trans_by[fam.name] = tr
        flatX_by[fam.name], flaty_by[fam.name] = Xf, yf
        print(f"[detect] {fam.name}: {time.time()-t0:.1f}s "
              f"dets={sum(len(v) for v in dets.values())} "
              f"transitions={len(tr)} train_pairs={len(yf)} pos={int(yf.sum())}",
              flush=True)

    # ---- Champion baseline CV (frozen motion champion, no descriptors) -------
    base_rows = []
    for fam in CV_HOLDOUT:
        pred = link_centroids(dets_by[fam.name], scale=scale_by[fam.name], params=champ_link)
        base_rows.append(score_family(fam, pred, gt_by[fam.name], ntrue_by[fam.name],
                                      scale=scale_by[fam.name]))
    baseline = aggregate(base_rows)
    base_adj = {r.name: r.adj_edge_jaccard for r in baseline.per_dataset}
    base_raw = {r.name: r.edge_jaccard for r in baseline.per_dataset}
    print(f"[baseline] champion micro_adj={baseline.micro_adj_edge_jaccard:.4f} "
          f"micro_raw={baseline.micro_edge_jaccard:.4f}", flush=True)

    # ---- LOFO: one fit per held-out family (model coeffs weight-independent) --
    fold_models, gaps = {}, {}
    for held in CV_HOLDOUT:
        t0 = time.time()
        train_tr = []
        for other in CV_HOLDOUT:
            if other.name != held.name:
                train_tr.extend(trans_by[other.name][::TRAIN_STRIDE])
        model = fit_cross_attention(
            train_tr, weight=0.0, d=MODEL_D, h2=MODEL_D, epochs=EPOCHS, seed=SEED
        )
        fold_models[held.name] = model
        Xf, yf = flatX_by[held.name], flaty_by[held.name]
        if len(Xf) and yf.sum() > 0 and (len(yf) - yf.sum()) > 0:
            # held-out discriminative gap: score the flat feature rows via a
            # single-pair forward (feasible=1) — a fair per-pair probability proxy.
            feas1 = np.ones((Xf.shape[0], 1), dtype=bool)
            planes1 = Xf[:, None, :]
            p = model.score_transition(planes1, feas1).ravel()
            gaps[held.name] = {
                "p_pos_mean": round(float(p[yf > 0.5].mean()), 4),
                "p_neg_mean": round(float(p[yf < 0.5].mean()), 4),
                "gap": round(float(p[yf > 0.5].mean() - p[yf < 0.5].mean()), 4),
                "n_pos": int(yf.sum()), "n_neg": int(len(yf) - yf.sum()),
            }
        print(f"[fit] held-out {held.name}: {time.time()-t0:.1f}s "
              f"gap={gaps.get(held.name, {}).get('gap')}", flush=True)

    # ---- Score the weight sweep (LOFO, leak-free) ----------------------------
    variants = []
    for w in WEIGHTS:
        rows = []
        for fam in CV_HOLDOUT:
            link = dataclasses.replace(
                champ_link, xattn_edge_model=fold_models[fam.name].with_weight(w).to_dict()
            )
            pred = link_centroids(dets_by[fam.name], scale=scale_by[fam.name],
                                  params=link, descriptors=descs_by[fam.name])
            rows.append(score_family(fam, pred, gt_by[fam.name], ntrue_by[fam.name],
                                     scale=scale_by[fam.name]))
        res = aggregate(rows)
        adj_by = {r.name: r.adj_edge_jaccard for r in res.per_dataset}
        raw_by = {r.name: r.edge_jaccard for r in res.per_dataset}
        no_reg_adj = all(adj_by[n] >= base_adj[n] - 1e-9 for n in base_adj)
        no_reg_raw = all(raw_by[n] >= base_raw[n] - 1e-9 for n in base_raw)
        d_tp = sum(r.edge_tp for r in res.per_dataset) - sum(r.edge_tp for r in baseline.per_dataset)
        d_fp = sum(r.edge_fp for r in res.per_dataset) - sum(r.edge_fp for r in baseline.per_dataset)
        d_fn = sum(r.edge_fn for r in res.per_dataset) - sum(r.edge_fn for r in baseline.per_dataset)
        beats = res.micro_adj_edge_jaccard > CHAMPION_REFERENCE_MICRO_ADJ + 1e-9
        row = {
            "weight": w,
            "micro_adj": round(res.micro_adj_edge_jaccard, 4),
            "micro_raw": round(res.micro_edge_jaccard, 4),
            "delta_micro_adj": round(res.micro_adj_edge_jaccard - baseline.micro_adj_edge_jaccard, 4),
            "no_regression_adj": bool(no_reg_adj),
            "no_regression_raw": bool(no_reg_raw),
            "total_edge_delta": {"d_tp": int(d_tp), "d_fp": int(d_fp), "d_fn": int(d_fn)},
            "per_dataset_adj": {k: round(v, 4) for k, v in adj_by.items()},
            "per_dataset_delta_adj": {n: round(adj_by[n] - base_adj[n], 4) for n in base_adj},
            "promotable": bool(no_reg_adj and no_reg_raw and beats),
            "macro_adj": round(res.macro_adj_edge_jaccard, 4),
            "lineage_macro_adj": round(res.lineage_macro_adj, 4),
        }
        variants.append(row)
        print(f"[weight] w={w:.1f} micro_adj={row['micro_adj']} "
              f"d={row['delta_micro_adj']:+} dTP={d_tp:+d} dFP={d_fp:+d} dFN={d_fn:+d} "
              f"no_reg_adj={no_reg_adj} no_reg_raw={no_reg_raw} "
              f"promotable={row['promotable']}", flush=True)

    zero = next(v for v in variants if v["weight"] == 0.0)
    sanity_reproduces = (
        abs(zero["micro_adj"] - round(baseline.micro_adj_edge_jaccard, 4)) < 1e-9
        and zero["total_edge_delta"] == {"d_tp": 0, "d_fp": 0, "d_fn": 0}
    )
    promotable = [v for v in variants if v["promotable"]]
    ranked = sorted(variants, key=lambda v: v["micro_adj"], reverse=True)
    mean_gap = (round(float(np.mean([g["gap"] for g in gaps.values()])), 4)
                if gaps else None)

    if promotable:
        verdict = "promoted"
    elif any(v["delta_micro_adj"] > 1e-9 and not (v["no_regression_adj"] and v["no_regression_raw"])
             for v in variants):
        verdict = "inconclusive"
    else:
        verdict = "rejected"

    champ_sha = _sha256(CHAMPION_CONFIG)
    payload = {
        "issue": "SOT-2994",
        "recordedAt": datetime.now(timezone.utc).isoformat(),
        "axis": (
            "learned cross-attention edge linking (SimpleNodeTransformer port): each "
            "edge embedding is contextualised by attention over its source's competing "
            "successors and its destination's competing predecessors before scoring "
            "p_edge; cost dist + w*(1-p_edge), <=max_distance motion gate unchanged "
            "(re-rank only, metric-valid); leak-free LOFO same-seed A/B vs the frozen "
            "ARGUS motion champion; linking-only ablation (detection frozen, cached)"
        ),
        "sources": [
            "royerlab SimpleNodeTransformer (official baseline cross-attention linker)",
            "SOT-2841 per-edge logistic re-rank (REJECTED, held-out gap 0.42, zero CV gain)",
            "SOT-2870 learned gate expansion (REJECTED, 44b6 regression)",
            "SOT-2864 ARGUS motion-model champion (current, micro_adj 0.6760)",
        ],
        "champion_config_sha256": champ_sha,
        "champion_byte_frozen": champ_sha == CHAMPION_CONFIG_SHA256,
        "champion_reference_micro_adj": CHAMPION_REFERENCE_MICRO_ADJ,
        "feature_names": list(EDGE_FEATURE_NAMES),
        "model": {"arch": "xattn-edge-v1", "d": MODEL_D, "h2": MODEL_D,
                  "epochs": EPOCHS, "seed": SEED, "train_transition_stride": TRAIN_STRIDE,
                  "train_float32_infer_numpy_float64": True,
                  "note": "training subsamples frame pairs by stride; full CV scored on every frame"},
        "detection": "champion (frozen, cached once/family) + descriptors; linking-only ablation",
        "sot2993_dependency": "none (handcrafted geometric+intensity fallback used; standalone-evaluable)",
        "baseline_champion": {
            "micro_adj": round(baseline.micro_adj_edge_jaccard, 4),
            "micro_raw": round(baseline.micro_edge_jaccard, 4),
            "per_dataset_adj": {k: round(v, 4) for k, v in base_adj.items()},
            "cv": cv_result_to_dict(baseline),
        },
        "weight_zero_sanity_reproduces_champion": bool(sanity_reproduces),
        "held_out_prob_gap": gaps,
        "mean_held_out_prob_gap": mean_gap,
        "grid": {"weight": WEIGHTS},
        "variants": variants,
        "variants_ranked_by_micro_adj": ranked,
        "n_promotable": len(promotable),
        "best_promotable": promotable[0] if promotable else None,
        "representativeness_champion": representativeness_report(baseline),
        "verdict": verdict,
        "kaggle_submission": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nVERDICT={verdict} sanity_reproduces_champion={sanity_reproduces} "
          f"mean_gap={mean_gap} n_promotable={len(promotable)}")
    print(f"champion_byte_frozen={payload['champion_byte_frozen']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
