"""Portable GT-learned edge-linking cost (SOT-2841).

The champion links each ``t -> t+1`` transition on **scaled centroid distance
alone**. In the dense ``6bba`` families several plausible successors sit within
``max_distance``, so the optimal-*distance* assignment attaches the wrong, merely
nearer neighbour — costing matched edges. Two earlier disambiguation axes were
REJECTED:

* **SOT-2828** learned a *node* scorer from the sparse GT — but the label space is
  positive-unlabeled (an unmatched candidate may be a real un-annotated cell), so a
  single threshold could not separate signal from noise across the family mix.
* **SOT-2829** added *one* hand-crafted appearance-similarity term to the link
  cost — a single non-learned feature, non-discriminating on its own.

This module is the untried **learned LINKING** axis the SOT-2828/2829 hand-offs
named explicitly: the supervision label is the **edge** (``label = edge``), a
*relative, dense* signal that is well-defined on the dense family (which of a
source's several distance-feasible successors is the GT one) and does **not** suffer
the node PU contamination — a feasible pair with a matched source has a definite GT
successor, so every other feasible successor of that source is a confident negative.

A light logistic classifier is fit **leak-free (leave-one-family-out)** on GT
*consecutive* edges (positives) vs. the feasible non-GT successors of matched
sources (negatives), over a joint **edge**-feature vector — scaled distance,
appearance-descriptor cosine (reusing the SOT-2829 patch descriptor), and the
local *competition* cues a dense cluster needs (how many rivals each endpoint has,
this pair's distance rank among the source's / destination's feasible partners, and
its distance margin over the source's nearest rival). The learned edge probability
``p_edge`` enters the link cost as ``dist + weight * (1 - p_edge)`` — **re-rank only,
metric-valid**: the ``<= max_distance`` feasibility gate stays on the raw scaled
distance, so the term only reorders the champion's existing feasible set and can
never introduce a new long-range (metric-invalid) edge.

**SOT-2870 — learned gate EXPANSION (motion + shape/intensity):** SOT-2841's re-rank
was CV-neutral because a fixed raw-distance feasible set is saturated (near ≈ GT). The
headroom SOT-2864 found (motion-model linking, +0.0111) is in the *feasibility gate*:
admitting a raw-far but **motion-consistent** successor the distance gate drops
(**FN-edge recovery**). This module's feature vector is therefore extended with the
SOT-2864 **motion residual** (predicted-position distance) and two Trackastra-style
shallow **shape/intensity ratios** (brightness / spread change, reused from the patch
descriptor — no new extraction, no train/infer skew). The learned ``p_edge`` then
drives a **gate-expansion admissibility** in the linker (:class:`LinkParams`
``edge_gate_expand``): a pair beyond ``max_distance`` in raw distance is admitted only
when it is motion-corrected in-range (``dist_pred <= max_distance``), within a bounded
raw ratio, **and** the classifier scores it a real edge — never an unbounded
long-range edge. Default-off (``edge_gate_expand=False``) keeps the champion
byte-for-byte.

Kernel-safe by construction (numpy/scipy only, embedded coefficients, no sklearn /
pickle / filesystem at inference, ``exec()`` / no ``__file__``):

* **Feature planes** (:func:`edge_feature_planes`) are pure numpy over the scaled
  distance matrix + standardised descriptors — the exact shared function training
  and inference both call, so there is no train/infer feature skew.
* **Inference** (:class:`LearnedEdgeCost`) is a single *standardize -> dot ->
  sigmoid* over the ``(P, Q)`` feature planes, with the fitted mean/std/coef
  embedded in the link config.

``weight == 0`` (or ``edge_cost_model is None``, or no descriptors) drops the term
entirely and reproduces the distance-only champion **byte-for-byte** (a strict
default-off superset).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Canonical, ordered edge-feature names. The serialized model records these so a
# config is self-describing and inference asserts the feature vector matches what it
# was fit on (a silent feature-order drift would corrupt the cost invisibly).
EDGE_FEATURE_NAMES: tuple[str, ...] = (
    "dist_scaled",     # scaled Euclidean centroid distance (microns)
    "app_cos",         # appearance-descriptor cosine similarity in [0, 1]
    "src_rivals",      # # of OTHER feasible successors of the source (row competition)
    "dst_rivals",      # # of OTHER feasible predecessors of the destination (col comp.)
    "succ_rank",       # 0-based distance rank of this dst among the source's feasibles
    "pred_rank",       # 0-based distance rank of this src among the dst's feasibles
    "succ_margin",     # dist - (source's nearest feasible successor distance) (>= 0)
    # --- SOT-2870 joint motion + shape/intensity features (gate-expansion axis) ---
    "motion_resid",    # scaled dist from the SOT-2864 motion-predicted src to dst
                       # (== dist_scaled when no motion field is supplied) — the
                       # residual that discriminates a motion-consistent successor
                       # (small) from a raw-near but motion-incoherent decoy (large).
    "intensity_ratio", # |src - dst| standardized mean-intensity (descriptor ch0):
                       # a real successor keeps a similar brightness frame-to-frame.
    "shape_ratio",     # |src - dst| standardized intensity-spread (descriptor ch1),
                       # a coarse size/shape-change proxy (Trackastra shallow cue).
)
N_EDGE_FEATURES: int = len(EDGE_FEATURE_NAMES)

# Descriptor channels reused for the shape/intensity ratio features (SOT-2870). The
# appearance descriptor (biohub_tracking.detect.patch_descriptors) packs the patch
# mean intensity at index 0 and its std (spread) at index 1 — reusing them keeps the
# joint edge feature train/infer-skew-free (no new extraction path).
_DESC_INTENSITY_CH = 0
_DESC_SPREAD_CH = 1


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    """Numerically-stable, warning-free logistic sigmoid (any array shape).

    Uses the ``logit >= 0`` / ``logit < 0`` split so the exponent is always ``<= 0``,
    and evaluates each branch only on its own side (no discarded overflowing
    ``exp(-logit)`` on the negative branch, so no spurious ``RuntimeWarning`` in a
    Kaggle kernel).
    """
    logit = np.asarray(logit, dtype=np.float64)
    out = np.empty_like(logit)
    pos = logit >= 0.0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-logit[pos]))
    ex = np.exp(logit[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def _appearance_cosine(src_desc: np.ndarray, dst_desc: np.ndarray) -> np.ndarray:
    """``(P, Q)`` appearance cosine similarity mapped to ``[0, 1]`` via ``(1+cos)/2``.

    Mirrors :func:`biohub_tracking.link._appearance_cost` so the learned term reuses
    the exact SOT-2829 descriptor similarity. A zero-norm descriptor (flat patch)
    yields neutral ``0.5``, never a divide-by-zero.
    """
    sn = np.linalg.norm(src_desc, axis=1)
    dn = np.linalg.norm(dst_desc, axis=1)
    denom = sn[:, None] * dn[None, :]
    dots = src_desc @ dst_desc.T
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where(denom > 1e-12, dots / denom, 0.0)
    cos = np.clip(cos, -1.0, 1.0)
    return 0.5 * (1.0 + cos)


def _abs_channel_diff(
    src_desc: np.ndarray, dst_desc: np.ndarray, channel: int, shape: tuple[int, int]
) -> np.ndarray:
    """``(P, Q)`` absolute difference of one descriptor channel (0 if absent).

    Used for the SOT-2870 shape/intensity ratio features. A descriptor with fewer
    than ``channel + 1`` columns (a degenerate/empty patch stat) yields a neutral
    all-zero plane rather than an index error, so the feature is always well defined.
    """
    p, q = shape
    if src_desc.ndim != 2 or dst_desc.ndim != 2:
        return np.zeros((p, q), dtype=np.float64)
    if src_desc.shape[1] <= channel or dst_desc.shape[1] <= channel:
        return np.zeros((p, q), dtype=np.float64)
    s = src_desc[:, channel].astype(np.float64)
    d = dst_desc[:, channel].astype(np.float64)
    return np.abs(s[:, None] - d[None, :])


def edge_feature_planes(
    dist_scaled: np.ndarray,
    src_desc: np.ndarray,
    dst_desc: np.ndarray,
    max_distance: float,
    dist_pred: np.ndarray | None = None,
) -> np.ndarray:
    """Edge-feature planes ``(P, Q, F)`` for one ``t -> t+1`` transition.

    ``dist_scaled`` is the ``(P, Q)`` scaled centroid distance matrix (the same one
    the linker gates on); ``src_desc``/``dst_desc`` are the aligned standardised
    appearance descriptors. Every feature is a pure function of the distance matrix
    and descriptors, so training and inference compute identical vectors. Infeasible
    pairs (``dist > max_distance``) still get a value but callers mask them out
    (their link cost is never consulted), so the ranks/margins/rival-counts are
    computed **only over the feasible set** — exactly the competition a dense
    cluster's assignment faces.

    ``dist_pred`` (SOT-2870, optional) is the ``(P, Q)`` scaled distance from each
    source's **motion-predicted** position (the SOT-2864 motion field) to each
    destination. When supplied, two things change so the joint feature vector matches
    the gate-*expansion* the model is trained/used for: (1) the ``motion_resid``
    feature carries the predicted-distance residual (raw distance otherwise), and
    (2) the feasible set the ranks/rivals/margins are computed over is the **union**
    of the raw ``<= max_distance`` set and the motion-corrected ``dist_pred <=
    max_distance`` set — i.e. the motion-admissible successors a gate-expanded
    assignment actually competes among. With ``dist_pred is None`` the function is
    byte-for-byte the original SOT-2841 behaviour (new columns are neutral).

    Feature order is :data:`EDGE_FEATURE_NAMES`.
    """
    p, q = dist_scaled.shape
    if dist_pred is None:
        feasible = dist_scaled <= max_distance
        motion_resid = dist_scaled.astype(np.float64)
    else:
        feasible = (dist_scaled <= max_distance) | (dist_pred <= max_distance)
        motion_resid = np.asarray(dist_pred, dtype=np.float64)
    inf = np.inf

    app = _appearance_cosine(src_desc, dst_desc)
    intensity_ratio = _abs_channel_diff(src_desc, dst_desc, _DESC_INTENSITY_CH, (p, q))
    shape_ratio = _abs_channel_diff(src_desc, dst_desc, _DESC_SPREAD_CH, (p, q))

    # Rival counts: number of OTHER feasible partners on each side (0 = the pair is
    # the source's/destination's only feasible option -> unambiguous).
    row_feas = feasible.sum(axis=1)  # (P,) feasible successors per source
    col_feas = feasible.sum(axis=0)  # (Q,) feasible predecessors per destination
    src_rivals = np.maximum(row_feas[:, None] - 1, 0).astype(np.float64)
    src_rivals = np.broadcast_to(src_rivals, (p, q))
    dst_rivals = np.maximum(col_feas[None, :] - 1, 0).astype(np.float64)
    dst_rivals = np.broadcast_to(dst_rivals, (p, q))

    # Distance ranks within the feasible set (infeasible entries pushed to the end
    # by masking to +inf before the double-argsort). rank 0 == nearest.
    masked = np.where(feasible, dist_scaled, inf)
    succ_rank = np.argsort(np.argsort(masked, axis=1), axis=1).astype(np.float64)
    pred_rank = np.argsort(np.argsort(masked, axis=0), axis=0).astype(np.float64)

    # Margin over the source's nearest feasible successor (0 for that nearest one;
    # a large margin means this successor is much farther than the best option).
    row_min = masked.min(axis=1)  # +inf for a source with no feasible successor
    row_min = np.where(np.isfinite(row_min), row_min, 0.0)
    succ_margin = np.maximum(dist_scaled - row_min[:, None], 0.0)

    planes = np.stack(
        [
            dist_scaled.astype(np.float64),
            app,
            src_rivals,
            dst_rivals,
            succ_rank,
            pred_rank,
            succ_margin,
            motion_resid,
            intensity_ratio,
            shape_ratio,
        ],
        axis=2,
    )
    assert planes.shape == (p, q, N_EDGE_FEATURES), planes.shape
    return planes


@dataclass(frozen=True)
class LearnedEdgeCost:
    """An embedded pure-numpy logistic edge-linking cost (SOT-2841).

    ``p_edge = sigmoid(((x - mean)/std) . coef + intercept)`` for each feasible
    ``(P, Q)`` edge-feature vector ``x`` (:func:`edge_feature_planes`); the added
    link cost is ``weight * (1 - p_edge)``. The fitted standardization + coefficients
    are stored in the link config so inference needs no sklearn / pickle / filesystem
    — a vectorised dot-product per transition (Kaggle-kernel safe).

    ``weight == 0`` makes the penalty identically zero, so the linker is byte-for-byte
    the distance-only champion.
    """

    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float
    weight: float

    def probability_planes(
        self,
        dist_scaled: np.ndarray,
        src_desc: np.ndarray,
        dst_desc: np.ndarray,
        max_distance: float,
        dist_pred: np.ndarray | None = None,
    ) -> np.ndarray:
        """``(P, Q)`` learned edge probability ``p_edge`` for every pair.

        ``dist_pred`` (SOT-2870) threads the motion-predicted distance into the joint
        feature vector so the same model scores both the re-rank (in-gate) and the
        gate-expansion (motion-corrected) admissibility decisions consistently.
        """
        planes = edge_feature_planes(
            dist_scaled, src_desc, dst_desc, max_distance, dist_pred=dist_pred
        )
        std = np.where(self.std > 1e-12, self.std, 1.0)
        z = (planes - self.mean) / std  # (P, Q, F)
        logit = z @ self.coef + self.intercept  # (P, Q)
        return _sigmoid(logit)

    def penalty(
        self,
        dist_scaled: np.ndarray,
        src_desc: np.ndarray,
        dst_desc: np.ndarray,
        max_distance: float,
        dist_pred: np.ndarray | None = None,
    ) -> np.ndarray:
        """``weight * (1 - p_edge)`` added-cost matrix ``(P, Q)`` (0 when weight 0)."""
        if self.weight == 0.0:
            return np.zeros_like(dist_scaled, dtype=np.float64)
        p_edge = self.probability_planes(
            dist_scaled, src_desc, dst_desc, max_distance, dist_pred=dist_pred
        )
        return self.weight * (1.0 - p_edge)

    def probability_rows(self, features: np.ndarray) -> np.ndarray:
        """``p_edge`` for an ``(N, F)`` stack of already-computed feature rows.

        Used by the leak-free screen to measure the held-out positive/negative
        probability gap (discriminative power), independent of the linking pass.
        """
        features = np.asarray(features, dtype=np.float64)
        if features.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        std = np.where(self.std > 1e-12, self.std, 1.0)
        z = (features - self.mean) / std
        logit = z @ self.coef + self.intercept
        return _sigmoid(logit)

    def to_dict(self) -> dict:
        """JSON-serialisable config block (embeddable in ``champion/config.json``)."""
        return {
            "feature_names": list(self.feature_names),
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "coef": [float(v) for v in self.coef],
            "intercept": float(self.intercept),
            "weight": float(self.weight),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LearnedEdgeCost":
        """Rebuild a model from its serialized config block (validates feature order)."""
        names = tuple(d["feature_names"])
        if names != EDGE_FEATURE_NAMES:
            raise ValueError(
                "edge_cost_model feature_names do not match the current "
                "EDGE_FEATURE_NAMES (feature extraction changed; retrain the model)"
            )
        return cls(
            feature_names=names,
            mean=np.asarray(d["mean"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
            coef=np.asarray(d["coef"], dtype=np.float64),
            intercept=float(d["intercept"]),
            weight=float(d.get("weight", 0.0)),
        )

    def with_weight(self, weight: float) -> "LearnedEdgeCost":
        """A copy at a different ``weight`` (the screen's re-rank-strength sweep)."""
        return LearnedEdgeCost(
            feature_names=self.feature_names,
            mean=self.mean,
            std=self.std,
            coef=self.coef,
            intercept=self.intercept,
            weight=float(weight),
        )


def _fit_logistic_numpy(
    z: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    l2: float,
    iters: int,
    lr: float,
) -> tuple[np.ndarray, float]:
    """Deterministic full-batch gradient-descent logistic regression (numpy).

    ``z`` is already standardized ``(N, F)``. Zero-initialised weights (no RNG), a
    fixed learning rate and L2 penalty — fully reproducible (mirrors the SOT-2828
    detection-scorer fit). Returns ``(coef, intercept)``.
    """
    _n, f = z.shape
    coef = np.zeros(f, dtype=np.float64)
    intercept = 0.0
    w = sample_weight.astype(np.float64)
    wsum = float(w.sum()) if w.sum() > 0 else 1.0
    for _ in range(iters):
        logit = z @ coef + intercept
        p = _sigmoid(logit)
        err = (p - y) * w
        grad_coef = z.T @ err / wsum + l2 * coef
        grad_int = float(err.sum()) / wsum
        coef -= lr * grad_coef
        intercept -= lr * grad_int
    return coef, intercept


def fit_edge_cost(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    weight: float = 0.0,
    l2: float = 0.1,
    iters: int = 3000,
    lr: float = 1.0,
) -> LearnedEdgeCost:
    """Fit a :class:`LearnedEdgeCost` from edge ``features`` ``(N, F)`` and 0/1 labels.

    Standardizes the features (storing the train mean/std for inference), then fits a
    logistic regression — sklearn's ``LogisticRegression`` if importable (training
    convenience only), else the embedded deterministic numpy gradient descent. Class
    imbalance (far more feasible negatives than GT-edge positives) is handled by
    inverse class frequency. The returned model carries ``weight`` as its re-rank
    strength (``0`` => an inert, byte-invariant term).
    """
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std > 1e-12, std, 1.0)
    z = (X - mean) / std_safe

    pos = float(y.sum())
    neg = float(len(y) - pos)
    w_pos = (len(y) / (2.0 * pos)) if pos > 0 else 0.0
    w_neg = (len(y) / (2.0 * neg)) if neg > 0 else 0.0
    sample_weight = np.where(y > 0.5, w_pos, w_neg).astype(np.float64)

    try:  # sklearn only if present; training-time convenience, never at inference.
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(C=1.0 / max(l2, 1e-6), max_iter=1000, solver="lbfgs")
        clf.fit(z, y, sample_weight=sample_weight)
        coef = clf.coef_.ravel().astype(np.float64)
        intercept = float(clf.intercept_[0])
    except Exception:
        coef, intercept = _fit_logistic_numpy(
            z, y, sample_weight, l2=l2, iters=iters, lr=lr
        )

    return LearnedEdgeCost(
        feature_names=EDGE_FEATURE_NAMES,
        mean=mean,
        std=std_safe,
        coef=coef,
        intercept=intercept,
        weight=float(weight),
    )
