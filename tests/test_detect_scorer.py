"""Tests for the portable GT-learned detection scorer (SOT-2828)."""

from __future__ import annotations

import json

import numpy as np

from biohub_tracking.detect import DetectParams, detect_centroids, detect_centroids_with_meta
from biohub_tracking.detect_scorer import (
    FEATURE_NAMES,
    N_FEATURES,
    LearnedScorer,
    extract_candidate_features,
    features_for_volume,
    fit_scorer,
    label_candidates,
)
from biohub_tracking.graph import TrackingGraph


def _blob_volume() -> np.ndarray:
    """A small volume with a few bright Gaussian blobs over low noise."""
    rng = np.random.default_rng(0)
    vol = rng.normal(0.02, 0.005, size=(8, 40, 40)).astype(np.float32)
    for zc, yc, xc in [(4, 10, 10), (4, 10, 28), (4, 28, 12), (3, 30, 30)]:
        zz, yy, xx = np.ogrid[0:8, 0:40, 0:40]
        vol += (
            0.8
            * np.exp(-(((zz - zc) / 1.2) ** 2 + ((yy - yc) / 2.4) ** 2 + ((xx - xc) / 2.4) ** 2))
        ).astype(np.float32)
    return vol


CHAMPION = DetectParams(
    sigma_zyx=(1.0, 2.0, 2.0),
    background_sigma_zyx=(2.0, 6.0, 6.0),
    nms_size_zyx=(2, 5, 5),
    threshold_percentile=92.0,
    mad_k=3.0,
)


def test_feature_matrix_shape_and_determinism():
    vol = _blob_volume()
    coords = detect_centroids(vol, CHAMPION)
    assert coords.shape[0] > 0
    feats, names = features_for_volume(vol, coords, CHAMPION)
    assert names == FEATURE_NAMES
    assert feats.shape == (coords.shape[0], N_FEATURES)
    assert np.isfinite(feats).all()
    feats2, _ = features_for_volume(vol, coords, CHAMPION)
    assert np.array_equal(feats, feats2)  # deterministic, no RNG


def test_empty_coords_features():
    vol = _blob_volume()
    feats, names = features_for_volume(vol, np.zeros((0, 3)), CHAMPION)
    assert feats.shape == (0, N_FEATURES)
    assert names == FEATURE_NAMES


def test_scorer_off_is_byte_identical():
    """detect_scorer=None reproduces the champion coords byte-for-byte."""
    vol = _blob_volume()
    base = detect_centroids(vol, CHAMPION)
    off = detect_centroids(vol, CHAMPION)
    assert np.array_equal(base, off)


def test_scorer_keep_all_reproduces_champion():
    """A threshold-0 scorer keeps every candidate → coords unchanged."""
    vol = _blob_volume()
    base = detect_centroids(vol, CHAMPION)
    feats, _ = features_for_volume(vol, base, CHAMPION)
    # Any coefficients: probability is in [0,1], so threshold 0.0 keeps all rows.
    scorer = LearnedScorer(
        feature_names=FEATURE_NAMES,
        mean=np.zeros(N_FEATURES),
        std=np.ones(N_FEATURES),
        coef=np.zeros(N_FEATURES),
        intercept=0.0,
        select_kind="threshold",
        select_value=0.0,
    )
    params = DetectParams(
        sigma_zyx=CHAMPION.sigma_zyx,
        background_sigma_zyx=CHAMPION.background_sigma_zyx,
        nms_size_zyx=CHAMPION.nms_size_zyx,
        threshold_percentile=CHAMPION.threshold_percentile,
        mad_k=CHAMPION.mad_k,
        detect_scorer=scorer.to_dict(),
    )
    kept, meta = detect_centroids_with_meta(vol, params)
    assert np.array_equal(kept, base)
    assert meta["scorer_kept"] == meta["scorer_total"] == base.shape[0]


def test_scorer_threshold_high_drops_all():
    vol = _blob_volume()
    base = detect_centroids(vol, CHAMPION)
    scorer = LearnedScorer(
        feature_names=FEATURE_NAMES,
        mean=np.zeros(N_FEATURES),
        std=np.ones(N_FEATURES),
        coef=np.zeros(N_FEATURES),
        intercept=0.0,  # probability == 0.5 everywhere
        select_kind="threshold",
        select_value=0.9,  # 0.5 < 0.9 → drop all
    )
    params = DetectParams(
        sigma_zyx=CHAMPION.sigma_zyx,
        background_sigma_zyx=CHAMPION.background_sigma_zyx,
        nms_size_zyx=CHAMPION.nms_size_zyx,
        threshold_percentile=CHAMPION.threshold_percentile,
        mad_k=CHAMPION.mad_k,
        detect_scorer=scorer.to_dict(),
    )
    kept, meta = detect_centroids_with_meta(vol, params)
    assert kept.shape[0] == 0
    assert meta["scorer_kept"] == 0
    assert meta["scorer_total"] == base.shape[0]


def test_scorer_roundtrip():
    scorer = LearnedScorer(
        feature_names=FEATURE_NAMES,
        mean=np.arange(N_FEATURES, dtype=float),
        std=np.arange(1, N_FEATURES + 1, dtype=float),
        coef=np.linspace(-1, 1, N_FEATURES),
        intercept=0.3,
        select_kind="threshold",
        select_value=0.4,
    )
    back = LearnedScorer.from_dict(scorer.to_dict())
    assert back.feature_names == FEATURE_NAMES
    assert np.allclose(back.mean, scorer.mean)
    assert np.allclose(back.std, scorer.std)
    assert np.allclose(back.coef, scorer.coef)
    assert back.intercept == scorer.intercept
    assert back.select_value == scorer.select_value
    feats = np.random.default_rng(1).normal(size=(5, N_FEATURES))
    assert np.allclose(scorer.probability(feats), back.probability(feats))


def test_fit_scorer_separates_synthetic():
    """A logistic fit on a linearly-separable feature learns to split it."""
    rng = np.random.default_rng(2)
    n = 400
    y = (rng.random(n) < 0.3).astype(float)
    feats = rng.normal(size=(n, N_FEATURES))
    feats[:, 0] += 4.0 * y  # feature 0 is the discriminative signal
    scorer = fit_scorer(feats, y)
    p = scorer.probability(feats)
    # Ranking AUC-ish: positives score higher on average than negatives.
    assert p[y > 0.5].mean() > p[y < 0.5].mean() + 0.3
    assert scorer.feature_names == FEATURE_NAMES


def test_from_dict_rejects_feature_drift():
    d = {
        "feature_names": ["wrong", "names"],
        "mean": [0.0, 0.0],
        "std": [1.0, 1.0],
        "coef": [0.0, 0.0],
        "intercept": 0.0,
        "select": ["threshold", 0.5],
    }
    try:
        LearnedScorer.from_dict(d)
    except ValueError:
        return
    raise AssertionError("expected ValueError on feature_names drift")


def test_scorer_embeds_in_config_and_runs_exec_compat():
    """A config-embedded scorer resolves via champion_params and runs with no
    filesystem / no __file__ (Kaggle-kernel constraint) — pure-numpy inference."""
    from biohub_tracking.champion import EMBEDDED_CHAMPION_CONFIG, champion_params

    scorer = LearnedScorer(
        feature_names=FEATURE_NAMES,
        mean=np.zeros(N_FEATURES),
        std=np.ones(N_FEATURES),
        coef=np.zeros(N_FEATURES),
        intercept=5.0,  # sigmoid(5) ~ 0.993 → keep at any reasonable threshold
        select_kind="threshold",
        select_value=0.5,
    )
    cfg = json.loads(json.dumps(EMBEDDED_CHAMPION_CONFIG))
    cfg["detect"]["detect_scorer"] = scorer.to_dict()
    detect_params, _link, _scale = champion_params(cfg)
    assert detect_params.detect_scorer is not None

    vol = _blob_volume()
    # exec() with no __file__ bound in globals — mirrors the submission kernel.
    ns: dict = {"vol": vol, "params": detect_params}
    code = (
        "from biohub_tracking.detect import detect_centroids_with_meta\n"
        "coords, meta = detect_centroids_with_meta(vol, params)\n"
    )
    exec(compile(code, "<string>", "exec"), ns)
    assert ns["meta"]["scorer_total"] >= ns["meta"]["scorer_kept"]
    # intercept 5 keeps ~everything → same as champion coords here.
    assert ns["coords"].shape[0] == detect_centroids(vol, CHAMPION).shape[0]


def test_label_candidates_matches_gt_within_distance():
    scale = (1.625, 0.40625, 0.40625)
    gt = TrackingGraph()
    # One GT node at t=0, voxel (4,10,10).
    gt.add_node(1, 0, 4, 10, 10)
    coords_by_t = {
        0: np.array(
            [
                [4, 10, 10],   # exact match → positive
                [4, 10, 12],   # ~0.8µm away → positive
                [4, 30, 30],   # far → negative (unlabeled)
            ],
            dtype=float,
        )
    }
    y, w = label_candidates(coords_by_t, gt, scale, max_distance=7.0)
    assert list(y) == [1.0, 1.0, 0.0]
    # PU weight: positives full, unlabeled negative down-weighted.
    assert w[0] == 1.0 and w[1] == 1.0
    assert 0.0 < w[2] < 1.0
