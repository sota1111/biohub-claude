"""Unit tests for intensity-weighted sub-voxel centroid refinement (SOT-3014).

The lever is a *count-neutral* detection post-step: it moves each kept centroid to
the centre-of-mass of the local normalized intensity without adding or removing
any point. The tests pin (1) exact byte-identical reproduction of the champion
when the knob is off, (2) count invariance when it is on, and (3) that a blob whose
true centre sits between voxels is refined toward that sub-voxel centre.
"""

from __future__ import annotations

import numpy as np

from biohub_tracking.detect import (
    DetectParams,
    _subvoxel_refine_coords,
    detect_centroids,
)


def _blob_volume(centers, shape=(16, 64, 64), amp=500.0, sigma=(1.0, 2.0, 2.0)):
    """(Z, Y, X) volume with a bright anisotropic Gaussian blob at each center."""
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


# The single-scale champion DoG config (base for the None-reproduction check).
_CHAMP = DetectParams(
    sigma_zyx=(1.0, 2.0, 2.0),
    background_sigma_zyx=(2.0, 6.0, 6.0),
    nms_size_zyx=(2, 5, 5),
    threshold_percentile=92.0,
    mad_k=3.0,
)


def test_none_reproduces_detector_exactly():
    """subvoxel_refine=None must be bit-identical to the champion detector."""
    vol = _blob_volume([(8, 16, 16), (8, 48, 48), (4, 32, 20)])
    base = detect_centroids(vol, _CHAMP)
    with_default = detect_centroids(
        vol, DetectParams(**{**_CHAMP.__dict__, "subvoxel_refine": None})
    )
    np.testing.assert_array_equal(base, with_default)
    # champion coords are integer voxel positions
    np.testing.assert_array_equal(base, np.round(base))


def test_refine_is_count_neutral():
    """Refinement moves centroids but never adds or removes any."""
    vol = _blob_volume([(8, 16, 16), (8, 48, 48), (4, 32, 20)])
    base = detect_centroids(vol, _CHAMP)
    refined = detect_centroids(
        vol, DetectParams(**{**_CHAMP.__dict__, "subvoxel_refine": (2, 5, 5)})
    )
    assert refined.shape == base.shape  # identical count
    # same set of centroids, only nudged (perfectly-centered synthetic blobs
    # stay put; the off-center pull is exercised in the dedicated test below)
    np.testing.assert_allclose(refined, base, atol=0.5)


def test_refine_pulls_toward_offcenter_mass():
    """A single voxel at an integer peak is pulled toward a nearby brighter voxel."""
    vol = np.zeros((8, 16, 16), dtype=np.float32)
    # integer NMS peak at (4, 8, 8); extra bright mass one voxel toward +x
    vol[4, 8, 8] = 10.0
    vol[4, 8, 9] = 6.0
    coords = np.array([[4.0, 8.0, 8.0]])
    refined = _subvoxel_refine_coords(vol, coords, (1, 2, 2))
    assert refined.shape == coords.shape
    # z, y unchanged; x pulled toward +x (into (8, 9))
    assert abs(refined[0, 0] - 4.0) < 1e-9
    assert abs(refined[0, 1] - 8.0) < 1e-9
    assert 8.0 < refined[0, 2] <= 9.0


def test_refine_flat_window_keeps_integer_centroid():
    """A degenerate all-equal window falls back to the original integer centroid."""
    vol = np.full((8, 16, 16), 3.0, dtype=np.float32)
    coords = np.array([[4.0, 8.0, 8.0]])
    refined = _subvoxel_refine_coords(vol, coords, (1, 2, 2))
    np.testing.assert_array_equal(refined, coords)


def test_refine_empty_is_noop():
    empty = np.zeros((0, 3))
    out = _subvoxel_refine_coords(np.zeros((4, 4, 4), dtype=np.float32), empty, (1, 2, 2))
    assert out.shape == (0, 3)
