"""Unit tests for per-volume robust quantile intensity normalization (SOT-2776)."""

from __future__ import annotations

import numpy as np

from biohub_tracking.detect import DetectParams, _normalize_intensity, detect_centroids


def _blob_volume() -> np.ndarray:
    """A small (Z, Y, X) volume with a smooth intensity drift + bright blobs.

    The linear background gradient (like the embryo brightness drift) makes p1 and
    p99 genuinely differ so the quantile rescale is exercised, not short-circuited
    by the degenerate-band guard.
    """
    z, y, x = np.mgrid[0:6, 0:24, 0:24]
    vol = (80.0 + 2.0 * x + 1.5 * y).astype(np.float32)  # smooth drift
    vol[3, 6, 6] += 400.0
    vol[3, 6, 5] += 200.0
    vol[2, 18, 17] += 300.0
    vol[3, 18, 18] += 150.0
    return vol


def test_none_spec_is_identity() -> None:
    vol = _blob_volume()
    out = _normalize_intensity(vol, None)
    assert out is vol  # untouched, same object


def test_quantile_clips_and_rescales_to_unit_range() -> None:
    vol = _blob_volume()
    out = _normalize_intensity(vol, ("quantile", 1.0, 99.0))
    assert out.dtype == np.float32
    assert out.min() == 0.0
    assert out.max() == 1.0
    # Voxels below the p1 floor / above the p99 ceiling saturate, not overflow.
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_degenerate_constant_band_left_unchanged() -> None:
    vol = np.zeros((3, 4, 4), dtype=np.float32)
    out = _normalize_intensity(vol, ("quantile", 1.0, 99.0))
    assert np.array_equal(out, vol)  # hi <= lo → no rescale, no NaN


def test_unknown_kind_raises() -> None:
    try:
        _normalize_intensity(np.zeros((2, 2, 2), np.float32), ("bogus", 1.0, 99.0))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown intensity_norm kind")


def test_default_detector_reproduces_without_norm() -> None:
    """intensity_norm=None must reproduce the pre-SOT-2776 detector exactly."""
    vol = _blob_volume()
    base = detect_centroids(vol, DetectParams(mad_k=3.0, background_sigma_zyx=(2.0, 6.0, 6.0)))
    same = detect_centroids(
        vol,
        DetectParams(mad_k=3.0, background_sigma_zyx=(2.0, 6.0, 6.0), intensity_norm=None),
    )
    assert np.array_equal(base, same)


def test_detector_runs_with_quantile_norm() -> None:
    vol = _blob_volume()
    params = DetectParams(
        mad_k=3.0,
        background_sigma_zyx=(2.0, 6.0, 6.0),
        intensity_norm=("quantile", 1.0, 99.0),
    )
    coords = detect_centroids(vol, params)
    assert coords.ndim == 2 and coords.shape[1] == 3
