"""Unit tests for the Hessian-eigenvalue blobness precision filter (SOT-2793)."""

from __future__ import annotations

import numpy as np
import pytest

from biohub_tracking.detect import (
    DetectParams,
    _hessian_blobness_mask,
    detect_centroids,
)


def _blob_volume(centers, shape=(16, 64, 64), amp=500.0, sigma=(1.0, 2.0, 2.0)):
    """(Z, Y, X) volume with a bright anisotropic Gaussian *blob* at each center."""
    zz, yy, xx = np.indices(shape)
    vol = np.full(shape, 50.0, dtype=np.float32)
    for cz, cy, cx in centers:
        vol += amp * np.exp(
            -(
                ((zz - cz) / sigma[0]) ** 2
                + ((yy - cy) / sigma[1]) ** 2
                + ((xx - cx) / sigma[2]) ** 2
            )
            / 2.0
        )
    return vol


def _line_volume(shape=(16, 64, 64), amp=500.0, sigma=(2.0, 2.0)):
    """(Z, Y, X) volume with one bright ridge running along X (a line/tube).

    Constant along X, a 2D Gaussian cross-section in (Z, Y) — the canonical
    non-blob (one Hessian eigenvalue ≈ 0 along the ridge axis).
    """
    zz, yy, _xx = np.indices(shape)
    cz, cy = shape[0] / 2, shape[1] / 2
    cross = amp * np.exp(
        -(((zz - cz) / sigma[0]) ** 2 + ((yy - cy) / sigma[1]) ** 2) / 2.0
    )
    return (50.0 + cross).astype(np.float32)


# The single-scale champion DoG config (base for the None-reproduction check).
_CHAMP = DetectParams(
    sigma_zyx=(1.0, 2.0, 2.0),
    background_sigma_zyx=(2.0, 6.0, 6.0),
    nms_size_zyx=(2, 5, 5),
    threshold_percentile=92.0,
    mad_k=3.0,
)


def test_none_reproduces_detector_exactly():
    """blobness_filter=None must be bit-identical to the unfiltered detector."""
    vol = _blob_volume([(8, 16, 16), (8, 48, 48), (4, 32, 20)])
    base = detect_centroids(vol, _CHAMP)
    with_default = detect_centroids(
        vol, DetectParams(**{**_CHAMP.__dict__, "blobness_filter": None})
    )
    assert np.array_equal(base, with_default)


def test_blob_survives_filter():
    """A genuine isotropic blob passes an aggressive blobness gate."""
    vol = _blob_volume([(8, 32, 32)])
    params = DetectParams(
        **{
            **_CHAMP.__dict__,
            "blobness_filter": ("hessian", (1.0, 2.0, 2.0), 0.05, 20.0),
        }
    )
    coords = detect_centroids(vol, params)
    assert len(coords) >= 1
    d = np.sqrt(((coords - np.array([8, 32, 32])) ** 2).sum(axis=1))
    assert d.min() <= 2.0


def test_line_is_pruned_but_blob_kept():
    """The blobness gate drops a line/ridge candidate while keeping a real blob."""
    line = _line_volume()
    blob = _blob_volume([(8, 48, 48)]) - 50.0  # add a blob far from the ridge
    vol = (line + blob).astype(np.float32)
    params_off = DetectParams(**{**_CHAMP.__dict__, "blobness_filter": None})
    params_on = DetectParams(
        **{
            **_CHAMP.__dict__,
            # require reasonable isotropy so the near-zero-eigenvalue ridge fails
            "blobness_filter": ("hessian", (1.0, 2.0, 2.0), 0.15, 8.0),
        }
    )
    coords_off = detect_centroids(vol, params_off)
    coords_on = detect_centroids(vol, params_on)
    # The blob is still detected...
    d_blob = np.sqrt(((coords_on - np.array([8, 48, 48])) ** 2).sum(axis=1))
    assert d_blob.min() <= 2.0
    # ...but the filter removed at least one (the ridge) candidate.
    assert len(coords_on) < len(coords_off)
    # No surviving centroid sits on the ridge core (z≈8, y≈32, any x).
    on_ridge = (np.abs(coords_on[:, 0] - 8) <= 1) & (np.abs(coords_on[:, 1] - 32) <= 1)
    assert not on_ridge.any()


def test_mask_all_negative_and_ratio_gates():
    """_hessian_blobness_mask enforces sign + isotropy + planarity directly."""
    vol = _blob_volume([(8, 32, 32)])
    coords = np.array([[8, 32, 32]], dtype=np.float64)
    # Permissive gate → blob kept.
    keep = _hessian_blobness_mask(vol, coords, ("hessian", (1.0, 2.0, 2.0), 0.01, 100.0))
    assert bool(keep[0])
    # Impossible isotropy requirement (min_blobness=1.5 > 1) → always dropped.
    keep2 = _hessian_blobness_mask(vol, coords, ("hessian", (1.0, 2.0, 2.0), 1.5, 100.0))
    assert not bool(keep2[0])


def test_mask_empty_candidates():
    vol = _blob_volume([(8, 32, 32)])
    keep = _hessian_blobness_mask(
        vol, np.zeros((0, 3)), ("hessian", (1.0, 2.0, 2.0), 0.1, 10.0)
    )
    assert keep.shape == (0,)
    assert keep.dtype == bool


def test_unknown_kind_raises():
    vol = _blob_volume([(8, 32, 32)])
    with pytest.raises(ValueError):
        _hessian_blobness_mask(
            vol, np.array([[8, 32, 32]], dtype=float), ("frangi", (1.0, 2.0, 2.0), 0.1, 10.0)
        )


def test_deterministic():
    vol = _blob_volume([(8, 20, 20), (6, 40, 44)])
    params = DetectParams(
        **{
            **_CHAMP.__dict__,
            "blobness_filter": ("hessian", (1.0, 2.0, 2.0), 0.1, 10.0),
        }
    )
    a = detect_centroids(vol, params)
    b = detect_centroids(vol, params)
    assert np.array_equal(a, b)
