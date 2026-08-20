"""Unit tests for multi-scale scale-space DoG blob detection (SOT-2774)."""

from __future__ import annotations

import numpy as np
import pytest

from biohub_tracking.detect import (
    DetectParams,
    _multiscale_dog_response,
    detect_centroids,
)


def _volume_with_blobs(centers, shape=(16, 64, 64), amp=500.0, sigma=(1.0, 2.0, 2.0)):
    """(Z, Y, X) volume with a bright anisotropic Gaussian blob at each center."""
    zz, yy, xx = np.indices(shape)
    vol = np.full(shape, 50.0, dtype=np.float32)
    for cz, cy, cx in centers:
        g = amp * np.exp(
            -(
                ((zz - cz) / sigma[0]) ** 2
                + ((yy - cy) / sigma[1]) ** 2
                + ((xx - cx) / sigma[2]) ** 2
            )
            / 2.0
        )
        vol += g
    return vol


def _nearest(coords, target):
    d = np.sqrt(((coords - np.asarray(target)) ** 2).sum(axis=1))
    return d.min()


# The single-scale champion DoG config (base for the None-reproduction check).
_SINGLE = DetectParams(
    sigma_zyx=(1.0, 2.0, 2.0),
    background_sigma_zyx=(2.0, 6.0, 6.0),
    nms_size_zyx=(2, 5, 5),
    threshold_percentile=92.0,
    mad_k=3.0,
)


def test_none_reproduces_single_scale_exactly():
    """dog_scales_zyx=None must be bit-identical to the single-scale detector."""
    vol = _volume_with_blobs([(8, 16, 16), (8, 48, 48), (4, 32, 20)])
    base = detect_centroids(vol, _SINGLE)
    with_default = detect_centroids(
        vol, DetectParams(**{**_SINGLE.__dict__, "dog_scales_zyx": None})
    )
    assert np.array_equal(base, with_default)


def test_middle_scale_matches_single_scale_response():
    """A bank whose middle scale == base sigma reproduces the champion response
    at that scale, so the combined-max response is >= the single-scale response."""
    vol = _volume_with_blobs([(8, 20, 20)])
    _, single_bg = None, None
    # Single-scale DoG response.
    from scipy import ndimage as ndi

    v = np.asarray(vol, dtype=np.float32)
    single = ndi.gaussian_filter(v, (1.0, 2.0, 2.0)) - ndi.gaussian_filter(
        v, (2.0, 6.0, 6.0)
    )
    # Multi-scale bank including the base scale, normalized by sigma^2 correction.
    params = DetectParams(
        sigma_zyx=(1.0, 2.0, 2.0),
        background_sigma_zyx=(2.0, 6.0, 6.0),
        dog_scales_zyx=((0.7, 1.4, 1.4), (1.0, 2.0, 2.0), (1.4, 2.8, 2.8)),
    )
    resp, scale_idx = _multiscale_dog_response(v, params)
    norm_mid = float(np.prod((1.0, 2.0, 2.0))) ** (2.0 / 3.0)
    # The combined max is at least the (normalized) middle-scale response everywhere.
    assert np.all(resp >= single * norm_mid - 1e-4)
    assert scale_idx.shape == v.shape
    assert set(np.unique(scale_idx)).issubset({0, 1, 2})


def test_multiscale_detects_varied_sizes():
    """A small and a large blob are both recovered by the multi-scale bank."""
    small = _volume_with_blobs([(8, 16, 16)], sigma=(1.0, 1.4, 1.4))
    large = _volume_with_blobs([(8, 48, 48)], sigma=(1.4, 3.0, 3.0)) - 50.0
    vol = small + large
    params = DetectParams(
        sigma_zyx=(1.0, 2.0, 2.0),
        background_sigma_zyx=(2.0, 6.0, 6.0),
        nms_size_zyx=(2, 5, 5),
        mad_k=3.0,
        dog_scales_zyx=((0.7, 1.4, 1.4), (1.0, 2.0, 2.0), (1.4, 2.8, 2.8)),
    )
    coords = detect_centroids(vol, params)
    assert _nearest(coords, (8, 16, 16)) <= 2.0
    assert _nearest(coords, (8, 48, 48)) <= 2.0


def test_deterministic():
    vol = _volume_with_blobs([(8, 20, 20), (6, 40, 44)])
    params = DetectParams(
        sigma_zyx=(1.0, 2.0, 2.0),
        background_sigma_zyx=(2.0, 6.0, 6.0),
        mad_k=3.0,
        dog_scales_zyx=((0.7, 1.4, 1.4), (1.0, 2.0, 2.0), (1.4, 2.8, 2.8)),
    )
    a = detect_centroids(vol, params)
    b = detect_centroids(vol, params)
    assert np.array_equal(a, b)


def test_requires_background_sigma():
    vol = _volume_with_blobs([(8, 20, 20)])
    params = DetectParams(
        sigma_zyx=(1.0, 2.0, 2.0),
        background_sigma_zyx=None,
        dog_scales_zyx=((1.0, 2.0, 2.0),),
    )
    with pytest.raises(ValueError):
        detect_centroids(vol, params)
