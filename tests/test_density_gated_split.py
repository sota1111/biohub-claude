"""Unit tests for density-gated nucleus splitting (SOT-2792).

The knob splits fused nuclei via marker-controlled watershed **only inside
locally dense regions**, so sparse families cannot fire it (the mechanistic fix
for SOT-2775's global-gate over-split). These tests pin the two invariants the
Issue requires: (1) ``density_gated_split=None`` reproduces the plain NMS
detector byte-for-byte, and (2) an isolated (sparse) blob field records **zero**
splits while a locally dense fused pair is split into both centroids.
"""

from __future__ import annotations

import numpy as np

from biohub_tracking.detect import (
    DetectParams,
    _local_peak_density,
    detect_centroids,
    detect_centroids_with_meta,
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

# A permissive split spec: h=0.5 robust-sigma, tiny min_size, no min_seed_dist,
# dense = at least 2 other detections within a 12-voxel window.
_SPEC = ("hmaxima", 0.5, 2.0, 0.0, 2.0, 12.0)


def test_none_reproduces_detector_exactly():
    """density_gated_split=None must be byte-identical to the plain detector."""
    vol = _blob_volume([(8, 16, 16), (8, 48, 48), (4, 32, 20)])
    base = detect_centroids(vol, _CHAMP)
    with_default = detect_centroids(
        vol, DetectParams(**{**_CHAMP.__dict__, "density_gated_split": None})
    )
    assert np.array_equal(base, with_default)


def test_meta_split_fired_zero_when_off():
    vol = _blob_volume([(8, 32, 32)])
    _coords, meta = detect_centroids_with_meta(vol, _CHAMP)
    assert meta["split_fired"] == 0


def test_local_peak_density_counts_neighbours():
    coords = np.array([[0, 0, 0], [0, 0, 3], [0, 0, 6], [0, 0, 40]], dtype=float)
    dens = _local_peak_density(coords, window=4.0)
    # first has one neighbour (the 3-away), middle has two, last is isolated.
    assert dens.tolist() == [1, 2, 1, 0]
    assert _local_peak_density(np.zeros((0, 3)), 4.0).shape == (0,)


def test_sparse_field_does_not_fire():
    """Far-apart (sparse) blobs stay below the density gate → zero splits, and the
    output is byte-identical to the plain NMS detector (SOT-2775 over-split
    avoided by construction)."""
    vol = _blob_volume([(8, 12, 12), (8, 12, 52), (8, 52, 12), (8, 52, 52)])
    nms = detect_centroids(vol, _CHAMP)
    gated, meta = detect_centroids_with_meta(
        vol, DetectParams(**{**_CHAMP.__dict__, "density_gated_split": _SPEC})
    )
    assert meta["split_fired"] == 0
    assert np.array_equal(nms, gated)


def test_dense_fused_pair_is_split():
    """Two touching nuclei that NMS merges into one peak are split when the region
    is locally dense (surrounded by other detections that raise the density).

    Uses a peak-resolving config (tight sigma, no broad DoG background) so the
    surrounding blobs each yield their own NMS centroid and raise the local
    density above the gate, while the ~5-voxel-apart pair still merges to one peak.
    """
    cfg = dict(
        sigma_zyx=(1.0, 1.5, 1.5),
        nms_size_zyx=(1, 3, 3),
        threshold_percentile=99.0,
        mad_k=3.0,
    )
    centers = [
        (8, 32, 30),
        (8, 32, 35),  # fused pair (~5 voxels apart → one NMS peak)
        (8, 26, 32),
        (8, 38, 32),
        (8, 32, 22),
        (8, 32, 44),
        (8, 26, 44),
        (8, 38, 22),
    ]
    vol = _blob_volume(centers, sigma=(1.0, 1.5, 1.5))
    spec = ("hmaxima", 0.3, 1.0, 0.0, 2.0, 16.0)
    base = detect_centroids(vol, DetectParams(**cfg))
    gated, meta = detect_centroids_with_meta(
        vol, DetectParams(**cfg, density_gated_split=spec)
    )
    # A dense component was split and both members of the fused pair are recovered.
    assert meta["split_fired"] >= 1
    assert len(gated) > len(base)
    near30 = np.any(np.abs(gated[:, 2] - 30) <= 2.0)
    near35 = np.any(np.abs(gated[:, 2] - 35) <= 2.0)
    assert near30 and near35
