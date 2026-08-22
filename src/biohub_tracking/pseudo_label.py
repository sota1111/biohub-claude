"""Semi-supervised **dense pseudo-label** self-training for the detection scorer
(SOT-2996).

Why this exists
---------------
The learned detection scorer (SOT-2828, :mod:`biohub_tracking.detect_scorer`) was
REJECTED because the sparse ground truth poisons its training set. The competition
GT annotates only *one* tracked lineage (~52 nodes) out of an estimated ~25 000
true cells per volume, so :func:`biohub_tracking.detect_scorer.label_candidates`
labels every candidate that does **not** match the sparse GT as a (down-weighted)
**negative** — including the thousands of *real but un-annotated* cells. Training on
that Positive-Unlabeled set teaches the scorer to rank real cells *low*: on the
SOT-2828 LOFO screen the held-out positive-minus-negative probability gap was
**negative on 3 of 4 families** (a real cell scored below noise). That is the
sparse-GT / density-mix wall in labeling form, and it is exactly the wall that node
budgeting (``J_adj = J·(1 − 0.1·(N_pred − N_true)/N_true)``) then punishes — the
sparse ``N_true`` makes any honest dense detection look like over-detection.

This module attacks the **root cause = sparse annotation** (escalation ladder
step-4 problem reformulation): instead of tuning an operating point, it *densifies
the teacher signal*. The classical champion detector's own high-confidence,
**temporally-consistent** detections become **dense pseudo-positive** labels for the
un-annotated cells, so the scorer is (re)trained on a densified label set rather
than a PU-poisoned one. A confidence + consistency gate suppresses false positives:

* **confidence** — the DoG response robust z-score (``resp_z``, the scorer's own
  first feature) must sit above a high percentile of the candidate pool.
* **temporal consistency** — the candidate must be absorbed into a champion-linked
  track of length ``>= min_track_len``. A bright blob that persists and *moves
  coherently* across frames is overwhelmingly a real nucleus; a transient bright
  speck (the dominant false-positive mode here) is not. Motion coherence is a GT-free
  signal, so the gate is leak-free.

Only a candidate that is (a) **not** matched to the sparse GT yet (b) passes *both*
gates is promoted to pseudo-positive; candidates that fail the gate become
**confident negatives** (full weight, no longer down-weighted as "maybe real"),
which is the second half of the PU repair.

Independence / non-blocking
---------------------------
The teacher bootstraps from the **classical champion** detector alone — it does not
require the SOT-2993 learned 3D-U-Net detector (which may or may not exist), so this
axis never hard-blocks on it. If a stronger learned detector is available its
high-confidence predictions can be fed in through the same ``resp_z`` /
``dets_by_t`` contract, but the classical bootstrap is the default and the tested
path.

Kernel-safe / default-off
--------------------------
Pure numpy/scipy (the champion linker is reused for the consistency gate), no torch,
no sklearn at inference, ``exec()``-safe. **Nothing here runs unless explicitly
called**: the champion pipeline never imports this module, so the byte-frozen
champion is unchanged. It produces a densified ``(labels, weights)`` pair that is a
drop-in replacement for :func:`biohub_tracking.detect_scorer.label_candidates` in a
scorer-training screen; it does not alter any promoted config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Row index of the DoG response robust z-score in the scorer feature vector
# (``biohub_tracking.detect_scorer.FEATURE_NAMES[0] == "resp_z"``). Asserted at
# call time against the passed feature-name tuple so a feature-order drift fails
# loudly instead of gating on the wrong column.
RESP_Z_FEATURE: str = "resp_z"


@dataclass(frozen=True)
class PseudoLabelConfig:
    """Gate parameters for dense pseudo-label promotion.

    resp_percentile
        Confidence gate: an unmatched candidate qualifies only if its ``resp_z`` is
        at or above this percentile of the *whole* candidate pool's ``resp_z``
        distribution (computed on the training pool passed to
        :func:`densify_labels`, so it is leak-free w.r.t. the held-out family).
    min_track_len
        Consistency gate: the candidate's champion-linked weakly-connected track
        must contain at least this many nodes. 1 disables the temporal gate.
    pseudo_weight
        Sample weight given to pseudo-positives (< 1.0: they are teacher labels, not
        ground truth, so they inform the fit less than a confidently-matched GT
        positive). :func:`biohub_tracking.detect_scorer.fit_scorer` multiplies this
        by inverse class frequency on top.
    neg_weight
        Sample weight for confident negatives (candidates that fail the gate). The
        SOT-2828 baseline down-weighted *all* unmatched candidates to 0.5 as
        "maybe real"; once the pseudo-gate has pulled the likely-real cells into the
        positive set the survivors are more trustworthy noise, so the default 1.0
        restores their full weight. Kept configurable for ablation.
    max_distance_um
        Node-match radius (microns) for the sparse-GT positive test — the exact 7 µm
        gate the competition matcher uses.
    """

    resp_percentile: float = 90.0
    min_track_len: int = 3
    pseudo_weight: float = 0.5
    neg_weight: float = 1.0
    max_distance_um: float = 7.0


def _sparse_gt_positive(
    dets_by_t: dict[int, np.ndarray],
    gt,
    scale: tuple[float, float, float],
    max_distance_um: float,
) -> np.ndarray:
    """0/1 sparse-GT match label per candidate, in ``row_t`` (sorted-t, det-order)
    order. Mirrors :func:`biohub_tracking.detect_scorer.label_candidates`'s match
    rule exactly (scaled-Euclidean 7 µm nearest-GT gate)."""
    from scipy.spatial import cKDTree

    scale_arr = np.asarray(scale, dtype=np.float64)
    gt_by_t = gt.nodes_by_time()
    out: list[np.ndarray] = []
    for t in sorted(dets_by_t):
        coords = np.asarray(dets_by_t[t], dtype=np.float64)
        if coords.shape[0] == 0:
            out.append(np.zeros(0, dtype=np.float64))
            continue
        gt_ids = gt_by_t.get(int(t), [])
        if not gt_ids:
            out.append(np.zeros(coords.shape[0], dtype=np.float64))
            continue
        gt_pos = np.stack([gt.position(g) for g in gt_ids]).astype(np.float64)
        tree = cKDTree(gt_pos * scale_arr)
        d, _ = tree.query(coords * scale_arr, k=1)
        out.append((d <= max_distance_um).astype(np.float64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


def track_length_per_candidate(
    dets_by_t: dict[int, np.ndarray],
    scale: tuple[float, float, float],
    link_params,
) -> np.ndarray:
    """Champion-linked weakly-connected track size for each candidate.

    Links *all* candidates with the champion linker and returns, per candidate (in
    ``row_t`` = sorted-t then det-order — the id order
    :func:`biohub_tracking.link.link_centroids` assigns), the number of nodes in the
    weakly-connected component (track / lineage fragment) it belongs to. This is the
    GT-free temporal-consistency signal: a real nucleus persists and moves
    coherently, forming a long component; a transient false detection stays a
    singleton.

    The champion linker's node-set mutators — short-track pruning
    (``min_track_length > 1``, which *drops* the very singletons this gate must see)
    and gap-recovery node interpolation (``gap_recover``, which *adds* synthetic
    nodes) — are disabled here so the graph keeps exactly one node per candidate, in
    ``row_t`` order (node id == row index, from
    :func:`biohub_tracking.link.link_centroids`'s dense id assignment). The
    *association* geometry (max distance, division, motion/velocity, etc.) is left at
    the champion's values, so the component sizes reflect the champion's linking. A
    pruned/interpolated graph would neither be 1:1 with the candidates nor retain the
    singletons the confidence+consistency gate is designed to reject.
    """
    from dataclasses import replace

    from .link import link_centroids

    # Disable only the two features that change the node *set* (add/remove nodes);
    # keep every association knob at the champion's value.
    consistency_params = replace(link_params, min_track_length=1, gap_recover=False)
    graph = link_centroids(dets_by_t, scale=scale, params=consistency_params)
    n = graph.num_nodes
    # Union-Find over undirected edges → component sizes, then map node id → size.
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for nid in graph.node_ids():
        for succ in graph.successors(nid):
            if 0 <= nid < n and 0 <= succ < n:
                union(nid, succ)
    sizes: dict[int, int] = {}
    for i in range(n):
        r = find(i)
        sizes[r] = sizes.get(r, 0) + 1
    # link_centroids assigns ids densely in row_t order, so node id == row index.
    return np.array([sizes[find(i)] for i in range(n)], dtype=np.int64)


@dataclass
class PseudoLabelDiagnostics:
    """False-positive / densification diagnostics for one label set."""

    n_candidates: int = 0
    n_gt_positive: int = 0
    n_pseudo_positive: int = 0
    n_confident_negative: int = 0
    # Recall of the gate on KNOWN-TRUE cells: of sparse-GT-matched candidates
    # (real cells), the fraction the confidence+consistency gate also accepts. High
    # = the gate does not throw away real cells (low false-negative on the teacher).
    gate_recall_on_gt: float = float("nan")
    # Consistency discrimination: mean track length of GT-positive candidates vs of
    # gate-rejected candidates. A wide gap means temporal consistency separates real
    # cells from transient noise (the FP-suppression signal).
    mean_tracklen_gt_positive: float = float("nan")
    mean_tracklen_rejected: float = float("nan")
    # Pseudo-positive FP proxy: of pseudo-positives that lie at a timepoint where the
    # sparse GT lineage IS annotated, the fraction that are NOT within the GT match
    # radius of any annotated node. This over-counts FP massively (the un-annotated
    # cells are real), so it is an *upper bound* on contamination, reported for
    # transparency rather than as a hard gate.
    pseudo_fp_upper_bound: float = float("nan")
    densification_ratio: float = float("nan")  # (gt_pos + pseudo_pos) / gt_pos

    def to_dict(self) -> dict:
        return {
            "n_candidates": int(self.n_candidates),
            "n_gt_positive": int(self.n_gt_positive),
            "n_pseudo_positive": int(self.n_pseudo_positive),
            "n_confident_negative": int(self.n_confident_negative),
            "gate_recall_on_gt": _round(self.gate_recall_on_gt),
            "mean_tracklen_gt_positive": _round(self.mean_tracklen_gt_positive),
            "mean_tracklen_rejected": _round(self.mean_tracklen_rejected),
            "pseudo_fp_upper_bound": _round(self.pseudo_fp_upper_bound),
            "densification_ratio": _round(self.densification_ratio),
        }


def _round(v: float, nd: int = 4) -> float:
    return round(float(v), nd) if not math.isnan(v) else float("nan")  # nan-safe


def densify_labels(
    dets_by_t: dict[int, np.ndarray],
    feats: np.ndarray,
    feature_names: tuple[str, ...],
    gt,
    scale: tuple[float, float, float],
    link_params,
    config: PseudoLabelConfig | None = None,
    *,
    resp_z_threshold: float | None = None,
    track_len: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, PseudoLabelDiagnostics]:
    """Dense pseudo-label a family's candidates.

    Returns ``(labels, weights, diagnostics)`` in the same ``row_t`` order
    (sorted-t, det-order) as :func:`biohub_tracking.detect_scorer.label_candidates`,
    so it is a drop-in replacement in a scorer-training screen. Labels:

    * **1** (weight 1.0) — sparse-GT matched candidate (ground-truth positive).
    * **1** (weight ``pseudo_weight``) — un-matched candidate that clears *both* the
      confidence gate (``resp_z >= resp_z_threshold``) and the consistency gate
      (champion track length ``>= min_track_len``). Dense pseudo-positive.
    * **0** (weight ``neg_weight``) — everything else (confident negative).

    ``resp_z_threshold`` may be supplied by the caller (e.g. computed once on the
    pooled training families so the gate is identical across LOFO folds and leak-free
    w.r.t. the held-out family); if ``None`` it is derived from *this* family's
    ``resp_z`` percentile. ``track_len`` may likewise be supplied pre-computed (the
    champion-linked component size per candidate, from
    :func:`track_length_per_candidate`) to avoid re-linking the same family across
    LOFO folds; if ``None`` it is computed here.
    """
    cfg = config or PseudoLabelConfig()
    if RESP_Z_FEATURE not in feature_names:
        raise ValueError(
            f"feature_names must contain {RESP_Z_FEATURE!r}; got {feature_names!r}"
        )
    resp_col = feature_names.index(RESP_Z_FEATURE)

    n = int(feats.shape[0]) if feats.ndim == 2 else 0
    y_gt = _sparse_gt_positive(dets_by_t, gt, scale, cfg.max_distance_um)
    if y_gt.shape[0] != n:
        raise ValueError(
            f"feature rows ({n}) != candidate rows ({y_gt.shape[0]}); "
            "feats must be built in row_t order over the same dets_by_t"
        )
    diag = PseudoLabelDiagnostics(n_candidates=n, n_gt_positive=int(y_gt.sum()))
    if n == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            diag,
        )

    resp_z = np.asarray(feats[:, resp_col], dtype=np.float64)
    if resp_z_threshold is None:
        resp_z_threshold = float(np.percentile(resp_z, cfg.resp_percentile))
    conf_gate = resp_z >= resp_z_threshold

    if track_len is None:
        if cfg.min_track_len > 1:
            track_len = track_length_per_candidate(dets_by_t, scale, link_params)
        else:
            track_len = np.full(n, cfg.min_track_len, dtype=np.int64)
    track_len = np.asarray(track_len, dtype=np.int64)
    if track_len.shape[0] != n:
        raise ValueError(
            f"track_len rows ({track_len.shape[0]}) != candidate rows ({n})"
        )
    consist_gate = track_len >= cfg.min_track_len

    is_gt = y_gt > 0.5
    # Pseudo-positive: unmatched AND high-confidence AND temporally consistent.
    pseudo = (~is_gt) & conf_gate & consist_gate

    labels = np.where(is_gt | pseudo, 1.0, 0.0)
    weights = np.where(
        is_gt, 1.0, np.where(pseudo, cfg.pseudo_weight, cfg.neg_weight)
    )

    # ---- diagnostics -------------------------------------------------------
    gate_accept = conf_gate & consist_gate
    diag.n_pseudo_positive = int(pseudo.sum())
    diag.n_confident_negative = int((labels < 0.5).sum())
    if is_gt.any():
        diag.gate_recall_on_gt = float(gate_accept[is_gt].mean())
        diag.mean_tracklen_gt_positive = float(track_len[is_gt].mean())
    rejected = ~gate_accept
    if rejected.any():
        diag.mean_tracklen_rejected = float(track_len[rejected].mean())
    diag.densification_ratio = (
        float((diag.n_gt_positive + diag.n_pseudo_positive) / diag.n_gt_positive)
        if diag.n_gt_positive
        else float("nan")
    )
    diag.pseudo_fp_upper_bound = _pseudo_fp_upper_bound(
        dets_by_t, pseudo, gt, scale, cfg.max_distance_um
    )
    return labels, weights, diag


def _pseudo_fp_upper_bound(
    dets_by_t: dict[int, np.ndarray],
    pseudo: np.ndarray,
    gt,
    scale: tuple[float, float, float],
    max_distance_um: float,
) -> float:
    """Upper bound on pseudo-positive false-positive contamination.

    Restricted to timepoints where the sparse GT lineage is annotated: of the
    pseudo-positives there, the fraction *not* within ``max_distance_um`` of any
    annotated node. Because the sparse GT annotates only one lineage, most real
    cells are unannotated and count here as "FP", so this is a deliberate **upper
    bound** — reported for transparency, never used as a hard gate.
    """
    from scipy.spatial import cKDTree

    scale_arr = np.asarray(scale, dtype=np.float64)
    gt_by_t = gt.nodes_by_time()
    # Walk candidates in the same row_t order used to build ``pseudo``.
    idx = 0
    n_eval = 0
    n_far = 0
    for t in sorted(dets_by_t):
        coords = np.asarray(dets_by_t[t], dtype=np.float64)
        m = coords.shape[0]
        if m == 0:
            continue
        sel = pseudo[idx : idx + m]
        idx += m
        gt_ids = gt_by_t.get(int(t), [])
        if not gt_ids or not sel.any():
            continue
        gt_pos = np.stack([gt.position(g) for g in gt_ids]).astype(np.float64)
        tree = cKDTree(gt_pos * scale_arr)
        d, _ = tree.query(coords[sel] * scale_arr, k=1)
        n_eval += int(sel.sum())
        n_far += int((d > max_distance_um).sum())
    return float(n_far / n_eval) if n_eval else float("nan")
