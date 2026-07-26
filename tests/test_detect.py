"""Detection on synthetic volumes with known blob centroids."""

from __future__ import annotations

import numpy as np

from biohub_tracking.detect import (
    DetectParams,
    detect_centroids,
    detect_volume_series,
)


def _volume_with_blobs(centers, shape=(16, 64, 64), amp=500.0, sigma=(1.0, 2.0, 2.0)):
    """Build a (Z, Y, X) volume with a bright Gaussian blob at each center."""
    zz, yy, xx = np.indices(shape)
    vol = np.full(shape, 50.0, dtype=np.float32)  # background
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


def test_detects_isolated_blobs():
    centers = [(8, 16, 16), (8, 48, 48), (4, 32, 20)]
    vol = _volume_with_blobs(centers)
    coords = detect_centroids(vol, DetectParams(threshold_percentile=99.0))
    assert len(coords) >= len(centers)
    # every planted blob has a detection within ~1.5 voxels
    for c in centers:
        assert _nearest(coords, c) <= 1.5


def test_returns_brightest_first():
    # one very bright + one dim blob; brightest must come first.
    vol = _volume_with_blobs([(8, 16, 16)], amp=2000.0)
    vol += _volume_with_blobs([(8, 48, 48)], amp=300.0) - 50.0
    coords = detect_centroids(vol, DetectParams(threshold_percentile=98.0))
    assert len(coords) >= 2
    assert _nearest(coords[:1], (8, 16, 16)) <= 1.5


def test_empty_when_flat():
    vol = np.full((8, 32, 32), 100.0, dtype=np.float32)
    coords = detect_centroids(vol, DetectParams(threshold_percentile=99.0))
    assert coords.shape == (0, 3)


def test_series_over_timepoints():
    vol0 = _volume_with_blobs([(8, 16, 16)])
    vol1 = _volume_with_blobs([(8, 18, 18)])
    arr = np.stack([vol0, vol1])  # (T, Z, Y, X)
    dets = detect_volume_series(arr, DetectParams(threshold_percentile=99.0))
    assert set(dets) == {0, 1}
    assert len(dets[0]) >= 1 and len(dets[1]) >= 1
