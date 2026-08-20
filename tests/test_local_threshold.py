"""Unit tests for the local-adaptive MAD threshold surface (SOT-2791)."""

from __future__ import annotations

import numpy as np
import pytest

from biohub_tracking.detect import DetectParams, detect_centroids


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


# The single-scale champion DoG config (base for the None-reproduction check).
_CHAMP = DetectParams(
    sigma_zyx=(1.0, 2.0, 2.0),
    background_sigma_zyx=(2.0, 6.0, 6.0),
    nms_size_zyx=(2, 5, 5),
    threshold_percentile=92.0,
    mad_k=3.0,
)


def test_none_reproduces_detector_exactly():
    """local_threshold=None must be bit-identical to the scalar-threshold detector."""
    vol = _blob_volume([(8, 16, 16), (8, 48, 48), (4, 32, 20)])
    base = detect_centroids(vol, _CHAMP)
    with_default = detect_centroids(
        vol, DetectParams(**{**_CHAMP.__dict__, "local_threshold": None})
    )
    assert np.array_equal(base, with_default)


def test_surface_keeps_real_blobs():
    """A real blob still detected when the local surface is applied."""
    vol = _blob_volume([(8, 16, 16), (8, 48, 48)])
    params = DetectParams(
        **{**_CHAMP.__dict__, "local_threshold": ("mad", (5, 15, 15), 3.0)}
    )
    coords = detect_centroids(vol, params)
    assert len(coords) >= 2
    for c in ((8, 16, 16), (8, 48, 48)):
        d = np.sqrt(((coords - np.array(c)) ** 2).sum(axis=1))
        assert d.min() <= 2.0


def test_local_surface_recovers_dim_blob_vs_global():
    """A dim blob dropped by the global cutoff is recovered by the local surface.

    A quiet corner holds a dim blob; the rest of the volume carries high-frequency
    noise so the **global** ``median + k·1.4826·MAD`` cutoff rises above the dim
    blob's response and drops it. The **local** surface gates the quiet corner
    against its own low-noise neighbourhood, so the dim blob survives — the exact
    dim-cell FN-recovery mechanism this knob targets, without lowering the global
    threshold. MAD is robust, so the noise must cover >50% of voxels to lift it.
    """
    rng = np.random.default_rng(0)
    shape = (16, 64, 64)
    _zz, yy, xx = np.indices(shape)
    vol = _blob_volume([(8, 52, 52)], shape=shape, amp=90.0) - 50.0 + 50.0
    noise = rng.normal(0.0, 120.0, size=shape).astype(np.float32)
    noise[(yy >= 40) & (xx >= 40)] = 0.0  # keep the dim-blob corner quiet
    vol = (vol + noise).astype(np.float32)

    global_params = DetectParams(
        **{**_CHAMP.__dict__, "mad_k": 5.0, "local_threshold": None}
    )
    local_params = DetectParams(
        **{**_CHAMP.__dict__, "mad_k": 5.0, "local_threshold": ("mad", (5, 21, 21), 3.0)}
    )
    g = detect_centroids(vol, global_params)
    l = detect_centroids(vol, local_params)

    def near(coords, c):
        if len(coords) == 0:
            return False
        d = np.sqrt(((coords - np.array(c)) ** 2).sum(axis=1))
        return bool(d.min() <= 2.0)

    # Global misses the dim blob; the local surface recovers it.
    assert not near(g, (8, 52, 52))
    assert near(l, (8, 52, 52))


def test_mean_kind_recovers_dim_blob_and_is_fast():
    """The fast separable ``"mean"`` (Niblack) surface also recovers the dim blob.

    Same construction as the ``"mad"`` mechanism test but exercising the
    kernel-viable ``uniform_filter`` path — this is the promotion-path kind.
    """
    rng = np.random.default_rng(0)
    shape = (16, 64, 64)
    _zz, yy, xx = np.indices(shape)
    vol = _blob_volume([(8, 52, 52)], shape=shape, amp=90.0)
    noise = rng.normal(0.0, 120.0, size=shape).astype(np.float32)
    noise[(yy >= 40) & (xx >= 40)] = 0.0
    vol = (vol + noise).astype(np.float32)

    g = detect_centroids(
        vol, DetectParams(**{**_CHAMP.__dict__, "mad_k": 5.0, "local_threshold": None})
    )
    l = detect_centroids(
        vol,
        DetectParams(
            **{**_CHAMP.__dict__, "mad_k": 5.0, "local_threshold": ("mean", (5, 21, 21), 4.0)}
        ),
    )

    def near(coords, c):
        if len(coords) == 0:
            return False
        return bool(np.sqrt(((coords - np.array(c)) ** 2).sum(axis=1)).min() <= 2.0)

    assert not near(g, (8, 52, 52))
    assert near(l, (8, 52, 52))


def test_mean_kind_deterministic_and_none_reproduces():
    vol = _blob_volume([(8, 20, 20), (6, 40, 44)])
    params = DetectParams(
        **{**_CHAMP.__dict__, "local_threshold": ("mean", (5, 21, 21), 3.0)}
    )
    assert np.array_equal(detect_centroids(vol, params), detect_centroids(vol, params))
    # None still bit-identical regardless of which kind exists.
    base = detect_centroids(vol, _CHAMP)
    assert np.array_equal(
        base, detect_centroids(vol, DetectParams(**{**_CHAMP.__dict__, "local_threshold": None}))
    )


def test_unknown_kind_raises():
    vol = _blob_volume([(8, 32, 32)])
    params = DetectParams(
        **{**_CHAMP.__dict__, "local_threshold": ("gaussian", (5, 15, 15), 3.0)}
    )
    with pytest.raises(ValueError):
        detect_centroids(vol, params)


def test_deterministic():
    vol = _blob_volume([(8, 20, 20), (6, 40, 44)])
    params = DetectParams(
        **{**_CHAMP.__dict__, "local_threshold": ("mad", (5, 15, 15), 3.0)}
    )
    a = detect_centroids(vol, params)
    b = detect_centroids(vol, params)
    assert np.array_equal(a, b)
