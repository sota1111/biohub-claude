"""Portable GT-learned detection scorer (SOT-2828).

Every detection *operating-point* lever tried on the ``detect-link-dog-v4-shorttrack``
champion — multiscale DoG (SOT-2774), quantile intensity normalization (SOT-2776),
marker-controlled watershed (SOT-2775), density-gated split (SOT-2792), local-MAD
threshold surface (SOT-2791), Hessian-blobness precision filter (SOT-2793) — was
REJECTED, because a single **global** image criterion cannot separate true nuclei
from over-detections on this data (the family-mix gate always fails: recovering the
dense ``6bba`` FN over-splits the sparse ``44b6`` family, and any FP-suppressing
threshold that helps ``6bba`` shatters ``44b6``'s clean detections).

This module is the untried **supervised** axis (the official royerlab baseline's
"GT-learned appearance detection" win, ported to portable classical ML): a light
classifier learned from the sparse ground truth that scores each NMS candidate on a
**joint** hand-crafted feature vector and re-ranks / selects the candidates to cut
false positives *without* a new global threshold. The learned model uses the
features **together** (a logistic decision surface in feature space), so it can — in
principle — separate signal from noise where no single feature's global operating
point could.

Kernel-safe by construction (design constraint: numpy/scipy only, offline, no GPU,
no weights, ``exec()`` / no ``__file__``):

* **Feature extraction** (:func:`extract_candidate_features`) is pure numpy/scipy —
  the DoG response strength, local intensity statistics (the SOT-2829 appearance
  patch descriptor), Hessian-of-Gaussian blobness eigen-ratios, and local neighbour
  density — the exact features the issue enumerates.
* **Inference** (:class:`LearnedScorer`) is a single *standardize → dot-product →
  sigmoid*, with the fitted coefficients **embedded in the champion config** (no
  sklearn, no pickle, no file at inference). Training (:func:`fit_scorer`) is a tiny
  deterministic numpy gradient-descent logistic regression (sklearn is used only if
  available, and only at *training* time), so the whole loop is dependency-light.

The scorer is applied by :func:`biohub_tracking.detect.detect_centroids_with_meta`
as a post-NMS keep-mask, default-off (``DetectParams.detect_scorer is None`` ⇒ the
champion detector is byte-for-byte unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

# Canonical, ordered feature names. The serialized scorer records these so a config
# is self-describing and inference asserts the feature vector matches what it was fit
# on (a silent feature-order drift would otherwise corrupt the score invisibly).
FEATURE_NAMES: tuple[str, ...] = (
    "resp_z",          # DoG response robust z-score at the candidate voxel
    "app_mean",        # local intensity patch: mean
    "app_std",         # local intensity patch: std
    "app_q10",         # local intensity patch: 10th percentile
    "app_q50",         # local intensity patch: median
    "app_q90",         # local intensity patch: 90th percentile
    "app_center",      # centre voxel intensity
    "app_contrast",    # centre - surround (blob contrast)
    "app_grad",        # mean |finite-difference| gradient (texture magnitude)
    "hess_isotropy",   # a0/a2 of |Hessian eigenvalues| (line/tube → 0)
    "hess_planarity",  # a2/a1 of |Hessian eigenvalues| (plane/membrane → large)
    "hess_logmag",     # log1p(sum |Hessian eigenvalues|) (blob curvature strength)
    "hess_allneg",     # 1.0 if all three eigenvalues < 0 (bright blob), else 0.0
    "local_density",   # # of other candidates within DENSITY_WINDOW voxels
)
N_FEATURES: int = len(FEATURE_NAMES)

# Neighbour-density window (voxels). Fixed (not scale-converted) so training and
# inference are identical regardless of the per-family voxel scale.
DENSITY_WINDOW: float = 8.0

# Planarity is a ratio that blows up for a perfectly planar structure (a1 ≈ 0); clip
# it to a finite cap so a degenerate candidate cannot dominate the standardization.
_PLANARITY_CAP: float = 1e3


def _hessian_eig_features(
    vol: np.ndarray, coords: np.ndarray, sigma_zyx: tuple[float, float, float]
) -> np.ndarray:
    """Per-candidate Hessian-of-Gaussian eigenvalue features (N, 4).

    Columns: ``[isotropy a0/a2, planarity a2/a1, log1p(sum|λ|), all_negative]`` with
    magnitudes ordered ``a0 ≤ a1 ≤ a2``. Mirrors
    :func:`biohub_tracking.detect._hessian_blobness_mask` (six 2nd-derivative
    Gaussian passes, eigen-decomposition only at the candidate voxels) but returns
    the continuous ratios as *features* rather than a hard keep/drop gate.
    """
    if coords.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float64)
    sigma = tuple(float(s) for s in sigma_zyx)
    hzz = ndi.gaussian_filter(vol, sigma, order=(2, 0, 0))
    hyy = ndi.gaussian_filter(vol, sigma, order=(0, 2, 0))
    hxx = ndi.gaussian_filter(vol, sigma, order=(0, 0, 2))
    hzy = ndi.gaussian_filter(vol, sigma, order=(1, 1, 0))
    hzx = ndi.gaussian_filter(vol, sigma, order=(1, 0, 1))
    hyx = ndi.gaussian_filter(vol, sigma, order=(0, 1, 1))

    idx = np.rint(coords).astype(np.intp)
    for a in range(3):
        idx[:, a] = np.clip(idx[:, a], 0, vol.shape[a] - 1)
    z, y, x = idx[:, 0], idx[:, 1], idx[:, 2]

    n = coords.shape[0]
    h = np.empty((n, 3, 3), dtype=np.float64)
    h[:, 0, 0] = hzz[z, y, x]
    h[:, 1, 1] = hyy[z, y, x]
    h[:, 2, 2] = hxx[z, y, x]
    h[:, 0, 1] = h[:, 1, 0] = hzy[z, y, x]
    h[:, 0, 2] = h[:, 2, 0] = hzx[z, y, x]
    h[:, 1, 2] = h[:, 2, 1] = hyx[z, y, x]

    ev = np.linalg.eigvalsh(h)  # (N, 3) ascending
    all_negative = np.all(ev < 0.0, axis=1).astype(np.float64)
    mag = np.sort(np.abs(ev), axis=1)  # a0 <= a1 <= a2
    a0, a1, a2 = mag[:, 0], mag[:, 1], mag[:, 2]
    isotropy = np.where(a2 > 0.0, a0 / a2, 0.0)
    planarity = np.where(a1 > 0.0, np.minimum(a2 / a1, _PLANARITY_CAP), _PLANARITY_CAP)
    logmag = np.log1p(a0 + a1 + a2)
    return np.stack([isotropy, planarity, logmag, all_negative], axis=1)


def extract_candidate_features(
    vol_normalized: np.ndarray,
    coords: np.ndarray,
    params,
    *,
    response: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Hand-crafted feature matrix ``(N, N_FEATURES)`` for detection candidates.

    ``vol_normalized`` is the volume the detector thresholds — i.e. **already**
    passed through :func:`biohub_tracking.detect._normalize_intensity` (so the
    caller inside ``detect_centroids_with_meta`` reuses the exact response). When
    ``response`` is ``None`` it is recomputed from ``vol_normalized`` via the shared
    :func:`biohub_tracking.detect._compute_response`, so a standalone caller
    (training / screen) gets the identical response the detector used.

    Deterministic, numpy/scipy-only, no RNG — Kaggle-kernel safe. Returns the matrix
    and :data:`FEATURE_NAMES` so callers can assert alignment.
    """
    from .detect import _compute_response, patch_descriptors

    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float64), FEATURE_NAMES

    if response is None:
        response = _compute_response(vol_normalized, params)

    # DoG response robust z-score at each candidate voxel (its detection strength
    # relative to the volume's own noise floor — the same robust scale mad_k uses).
    median = float(np.median(response))
    mad = float(np.median(np.abs(response - median)))
    robust_sigma = 1.4826 * mad
    idx = np.rint(coords).astype(np.intp)
    for a in range(3):
        idx[:, a] = np.clip(idx[:, a], 0, response.shape[a] - 1)
    resp_at = response[idx[:, 0], idx[:, 1], idx[:, 2]]
    resp_z = (resp_at - median) / (robust_sigma if robust_sigma > 0.0 else 1.0)

    # Local intensity statistics = the SOT-2829 appearance patch descriptor. Feed
    # the already-normalized volume with intensity_norm disabled so it is not
    # double-normalized (patch_descriptors normalizes internally otherwise).
    from .detect import DetectParams

    patch_params = DetectParams(
        sigma_zyx=params.sigma_zyx,
        nms_size_zyx=params.nms_size_zyx,
        intensity_norm=None,
    )
    app = patch_descriptors(vol_normalized, coords, params=patch_params)  # (N, 8)

    hess = _hessian_eig_features(vol_normalized, coords, params.sigma_zyx)  # (N, 4)

    # Local neighbour density: # of other candidates within DENSITY_WINDOW voxels.
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    counts = tree.query_ball_point(coords, r=DENSITY_WINDOW, return_length=True)
    density = (np.asarray(counts, dtype=np.float64) - 1.0).reshape(-1, 1)

    feats = np.concatenate(
        [resp_z.reshape(-1, 1), app, hess, density], axis=1
    )
    assert feats.shape[1] == N_FEATURES, (feats.shape, N_FEATURES)
    return feats.astype(np.float64, copy=False), FEATURE_NAMES


def features_for_volume(
    volume: np.ndarray, coords: np.ndarray, params
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Convenience: normalize a **raw** volume then extract candidate features.

    Used by training / screening where the raw ``(Z, Y, X)`` volume is on hand (the
    detector-internal caller passes the already-normalized volume + cached response
    to :func:`extract_candidate_features` directly).
    """
    from .detect import _normalize_intensity

    vol = _normalize_intensity(np.asarray(volume, dtype=np.float32), params.intensity_norm)
    return extract_candidate_features(vol, coords, params)


@dataclass(frozen=True)
class LearnedScorer:
    """An embedded pure-numpy logistic detection scorer (SOT-2828).

    ``probability(feats) = sigmoid(((feats - mean)/std) · coef + intercept)``. The
    fitted ``mean`` / ``std`` standardization and ``coef`` / ``intercept`` are stored
    in the config so inference needs no sklearn, no pickle, and no filesystem — a
    single vectorised dot-product per timepoint (Kaggle-kernel safe). ``select`` is
    the keep rule applied to the probabilities; only ``("threshold", p)`` is defined
    (keep candidates with ``probability >= p``).
    """

    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float
    select_kind: str = "threshold"
    select_value: float = 0.5

    def probability(self, feats: np.ndarray) -> np.ndarray:
        """Vectorised P(true nucleus) for each candidate feature row ``(N, F)``."""
        feats = np.asarray(feats, dtype=np.float64)
        if feats.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        if feats.shape[1] != self.coef.shape[0]:
            raise ValueError(
                f"feature dim {feats.shape[1]} != scorer dim {self.coef.shape[0]}"
            )
        std = np.where(self.std > 1e-12, self.std, 1.0)
        z = (feats - self.mean) / std
        logit = z @ self.coef + self.intercept
        # Numerically stable logistic.
        return np.where(
            logit >= 0.0,
            1.0 / (1.0 + np.exp(-logit)),
            np.exp(logit) / (1.0 + np.exp(logit)),
        )

    def keep_mask(self, feats: np.ndarray) -> np.ndarray:
        """Boolean keep-mask per candidate under the configured ``select`` rule."""
        if self.select_kind != "threshold":
            raise ValueError(f"unknown select kind: {self.select_kind!r}")
        p = self.probability(feats)
        return p >= float(self.select_value)

    def to_dict(self) -> dict:
        """JSON-serialisable config block (embeddable in ``champion/config.json``)."""
        return {
            "feature_names": list(self.feature_names),
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "coef": [float(v) for v in self.coef],
            "intercept": float(self.intercept),
            "select": [self.select_kind, float(self.select_value)],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LearnedScorer":
        """Rebuild a scorer from its serialized config block."""
        names = tuple(d["feature_names"])
        if tuple(names) != FEATURE_NAMES:
            raise ValueError(
                "scorer feature_names do not match the current FEATURE_NAMES "
                "(feature extraction changed; retrain the scorer)"
            )
        select = d.get("select", ["threshold", 0.5])
        return cls(
            feature_names=names,
            mean=np.asarray(d["mean"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
            coef=np.asarray(d["coef"], dtype=np.float64),
            intercept=float(d["intercept"]),
            select_kind=str(select[0]),
            select_value=float(select[1]),
        )

    def with_select(self, kind: str, value: float) -> "LearnedScorer":
        """A copy with a different selection rule (threshold sweep in a screen)."""
        return LearnedScorer(
            feature_names=self.feature_names,
            mean=self.mean,
            std=self.std,
            coef=self.coef,
            intercept=self.intercept,
            select_kind=kind,
            select_value=value,
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
    fixed learning rate and L2 penalty — fully reproducible. Returns ``(coef,
    intercept)``. Used only when sklearn is unavailable (the default here).
    """
    n, f = z.shape
    coef = np.zeros(f, dtype=np.float64)
    intercept = 0.0
    w = sample_weight.astype(np.float64)
    wsum = float(w.sum()) if w.sum() > 0 else 1.0
    for _ in range(iters):
        logit = z @ coef + intercept
        p = np.where(
            logit >= 0.0,
            1.0 / (1.0 + np.exp(-logit)),
            np.exp(logit) / (1.0 + np.exp(logit)),
        )
        err = (p - y) * w
        grad_coef = z.T @ err / wsum + l2 * coef
        grad_int = float(err.sum()) / wsum
        coef -= lr * grad_coef
        intercept -= lr * grad_int
    return coef, intercept


def fit_scorer(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray | None = None,
    *,
    l2: float = 0.1,
    iters: int = 3000,
    lr: float = 1.0,
    select_value: float = 0.5,
) -> LearnedScorer:
    """Fit a :class:`LearnedScorer` from candidate ``features`` and 0/1 ``labels``.

    Standardizes the features (storing the train mean/std for inference), then fits a
    logistic regression — sklearn's ``LogisticRegression`` if importable (training
    convenience only), else the embedded deterministic numpy gradient descent. The
    resulting coefficients are transformed back to operate on **standardized**
    features (the scorer standardizes at inference with the stored mean/std), so the
    embedded model is self-contained. Class imbalance is handled by
    ``sample_weight`` (default: inverse class frequency).
    """
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std > 1e-12, std, 1.0)
    z = (X - mean) / std_safe

    # Inverse class frequency so positives and (the far more numerous) negatives
    # contribute equally, multiplied by any per-sample PU weight the caller supplies
    # (:func:`label_candidates` down-weights unlabeled negatives).
    pos = float(y.sum())
    neg = float(len(y) - pos)
    w_pos = (len(y) / (2.0 * pos)) if pos > 0 else 0.0
    w_neg = (len(y) / (2.0 * neg)) if neg > 0 else 0.0
    class_w = np.where(y > 0.5, w_pos, w_neg)
    if sample_weight is None:
        sample_weight = class_w
    else:
        sample_weight = class_w * np.asarray(sample_weight, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)

    try:  # sklearn only if present; training-time convenience, never at inference.
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            C=1.0 / max(l2, 1e-6), max_iter=1000, solver="lbfgs"
        )
        clf.fit(z, y, sample_weight=sample_weight)
        coef = clf.coef_.ravel().astype(np.float64)
        intercept = float(clf.intercept_[0])
    except Exception:
        coef, intercept = _fit_logistic_numpy(
            z, y, sample_weight, l2=l2, iters=iters, lr=lr
        )

    return LearnedScorer(
        feature_names=FEATURE_NAMES,
        mean=mean,
        std=std_safe,
        coef=coef,
        intercept=intercept,
        select_kind="threshold",
        select_value=select_value,
    )


def label_candidates(
    coords_by_t: dict[int, np.ndarray],
    gt,
    scale: tuple[float, float, float],
    max_distance: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse-GT labels + PU sample weights for detection candidates.

    A candidate at timepoint ``t`` is a **positive** iff a GT node at ``t`` lies
    within ``max_distance`` microns (scaled Euclidean) — the exact node-match rule
    the competition matcher uses. Unmatched candidates are **unlabeled negatives**:
    the GT is sparse (only the tracked lineage is annotated), so an unmatched
    candidate may be a real but un-annotated cell rather than pure noise.

    Positive-unlabeled handling: every unmatched (unlabeled) candidate is
    down-weighted to ``neg_weight`` (< 1) so a possibly-real cell that got a spurious
    negative label contributes less to the fit than a confidently-matched positive;
    :func:`fit_scorer` multiplies this by inverse class frequency on top. Returns
    ``(labels 0/1, sample_weight)`` in the same row order as
    ``np.vstack([coords_by_t[t] for t in sorted(coords_by_t)])``.
    """
    from scipy.spatial import cKDTree

    scale_arr = np.asarray(scale, dtype=np.float64)
    gt_by_t = gt.nodes_by_time()

    labels: list[np.ndarray] = []
    for t in sorted(coords_by_t):
        coords = np.asarray(coords_by_t[t], dtype=np.float64)
        if coords.shape[0] == 0:
            labels.append(np.zeros(0, dtype=np.float64))
            continue
        gt_ids = gt_by_t.get(int(t), [])
        if not gt_ids:
            labels.append(np.zeros(coords.shape[0], dtype=np.float64))
            continue
        gt_pos = np.stack([gt.position(g) for g in gt_ids]).astype(np.float64)
        # Scale into microns so the 7µm gate matches the competition matcher.
        tree = cKDTree(gt_pos * scale_arr)
        d, _ = tree.query(coords * scale_arr, k=1)
        labels.append((d <= max_distance).astype(np.float64))

    y = np.concatenate(labels) if labels else np.zeros(0, dtype=np.float64)
    # PU weighting: matched positives full weight; unmatched (unlabeled) negatives
    # down-weighted, since some are un-annotated real cells (sparse GT). Class
    # balance is applied on top by fit_scorer's default inverse-frequency weight.
    neg_weight = 0.5
    base = np.where(y > 0.5, 1.0, neg_weight)
    return y, base
