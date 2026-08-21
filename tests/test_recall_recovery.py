"""Unit tests for recall-oriented FN-edge-endpoint recovery (SOT-2873).

The knob adds a bounded, strongest-first tier of *sub-threshold* local maxima to
raise GT-node recall @7 µm, exploiting the metric's no-node-FP property. It must
be byte-for-byte off by default, never touch the champion primary peaks, cap the
added peaks, and compose through the champion config plumbing.
"""

from __future__ import annotations

import numpy as np
import pytest

from biohub_tracking.champion import EMBEDDED_CHAMPION_CONFIG, champion_params
from biohub_tracking.detect import DetectParams, detect_centroids_with_meta


def _blob_volume(centers, amps, shape=(16, 64, 64), sigma=(1.0, 2.0, 2.0)):
    """(Z, Y, X) volume with an anisotropic Gaussian blob of ``amp`` at each center."""
    zz, yy, xx = np.indices(shape)
    vol = np.full(shape, 50.0, dtype=np.float32)
    for (cz, cy, cx), amp in zip(centers, amps):
        vol += amp * np.exp(
            -(
                ((zz - cz) / sigma[0]) ** 2
                + ((yy - cy) / sigma[1]) ** 2
                + ((xx - cx) / sigma[2]) ** 2
            )
            / 2.0
        )
    return vol


# The single-scale champion DoG config (adaptive mad_k path).
_CHAMP = DetectParams(
    sigma_zyx=(1.0, 2.0, 2.0),
    background_sigma_zyx=(2.0, 6.0, 6.0),
    nms_size_zyx=(2, 5, 5),
    threshold_percentile=92.0,
    mad_k=3.0,
)


def _with(**over):
    return DetectParams(**{**_CHAMP.__dict__, **over})


def test_none_reproduces_detector_exactly():
    """recall_recovery=None must be bit-identical to the champion detector."""
    vol = _blob_volume(
        [(8, 16, 16), (8, 48, 48), (4, 32, 20)], amps=[600.0, 400.0, 250.0]
    )
    base, _ = detect_centroids_with_meta(vol, _CHAMP)
    default, _ = detect_centroids_with_meta(vol, _with(recall_recovery=None))
    assert np.array_equal(base, default)


def test_zero_frac_is_a_noop():
    """max_extra_frac=0 adds nothing → champion coords byte-for-byte."""
    vol = _blob_volume([(8, 16, 16), (8, 48, 48)], amps=[600.0, 300.0])
    base, _ = detect_centroids_with_meta(vol, _CHAMP)
    same, meta = detect_centroids_with_meta(
        vol, _with(recall_recovery=("madk_tier", 1.0, 0.0))
    )
    assert np.array_equal(base, same)
    assert meta["recall_extra_kept"] == 0


def test_klow_at_or_above_madk_is_a_noop():
    """k_low >= mad_k leaves an empty band → nothing added."""
    vol = _blob_volume([(8, 16, 16), (8, 48, 48)], amps=[600.0, 300.0])
    base, _ = detect_centroids_with_meta(vol, _CHAMP)
    same, meta = detect_centroids_with_meta(
        vol, _with(recall_recovery=("madk_tier", 3.0, 1.0))
    )
    assert np.array_equal(base, same)
    assert meta["recall_extra_kept"] == 0


def test_recovery_appends_subthreshold_peaks_after_primary():
    """The recall tier only *appends*: primary is an exact prefix, extras are dimmer.

    A field of many weak blobs on top of a few bright ones guarantees the strict
    champion cutoff drops some genuine local maxima; the recall tier admits a
    bounded set of them. The champion coords must be a byte-for-byte prefix of the
    recovered coords, and every appended coord must sit below the champion cutoff
    (dimmer than every primary peak — so global brightest-first order holds).
    """
    rng = np.random.default_rng(1)
    centers = [(8, 12, 20), (8, 40, 40), (4, 24, 60)]
    vol = _blob_volume(centers, amps=[900.0, 850.0, 800.0], shape=(16, 80, 80))
    # A textured background creates many weak local-contrast maxima that sit in the
    # sub-threshold band, exactly the FN-endpoint candidates the tier is meant to
    # admit (mirrors the noisy real volumes, where sub-threshold peaks abound).
    vol = (vol + rng.normal(0.0, 20.0, size=vol.shape).astype(np.float32)).astype(
        np.float32
    )
    base, _ = detect_centroids_with_meta(vol, _CHAMP)
    rec, meta = detect_centroids_with_meta(
        vol, _with(recall_recovery=("madk_tier", 0.2, 1.0))
    )
    # Primary tier is untouched: base is an exact prefix of the recovered coords.
    assert np.array_equal(rec[: len(base)], base)
    assert meta["recall_primary"] == len(base)
    assert meta["recall_extra_kept"] == len(rec) - len(base)
    assert meta["recall_extra_kept"] >= 1  # some sub-threshold peak was admitted


def test_cap_limits_added_peaks():
    """The added tier never exceeds floor(max_extra_frac * n_primary)."""
    rng = np.random.default_rng(0)
    centers = [(8, 8 + 6 * i, 8 + 5 * (i % 8)) for i in range(20)]
    amps = [800.0] * 4 + list(rng.uniform(90.0, 150.0, size=16))
    vol = _blob_volume(centers, amps=amps, shape=(16, 64, 64))
    base, _ = detect_centroids_with_meta(vol, _CHAMP)
    n_primary = len(base)
    _, meta = detect_centroids_with_meta(
        vol, _with(recall_recovery=("madk_tier", 0.5, 0.25))
    )
    assert meta["recall_primary"] == n_primary
    assert meta["recall_extra_kept"] <= int(np.floor(0.25 * n_primary))
    # And when there ARE sub-threshold candidates, the cap is what binds.
    if meta["recall_extra_candidates"] > int(np.floor(0.25 * n_primary)):
        assert meta["recall_extra_kept"] == int(np.floor(0.25 * n_primary))


def test_requires_mad_k():
    """recall_recovery on a percentile (non-adaptive) config raises."""
    vol = _blob_volume([(8, 16, 16)], amps=[600.0])
    params = _with(mad_k=None, recall_recovery=("madk_tier", 1.0, 0.5))
    with pytest.raises(ValueError, match="mad_k"):
        detect_centroids_with_meta(vol, params)


def test_incompatible_with_local_threshold():
    """recall_recovery with a local_threshold surface raises (array gate)."""
    vol = _blob_volume([(8, 16, 16)], amps=[600.0])
    params = _with(
        local_threshold=("mean", (5, 15, 15), 3.0),
        recall_recovery=("madk_tier", 1.0, 0.5),
    )
    with pytest.raises(ValueError, match="local_threshold"):
        detect_centroids_with_meta(vol, params)


def test_champion_params_absent_key_is_off():
    """The embedded champion config carries no recall_recovery → None (byte-frozen)."""
    detect, _link, _scale = champion_params()
    assert detect.recall_recovery is None
    # Explicitly: the embedded config dict has no such key.
    assert "recall_recovery" not in EMBEDDED_CHAMPION_CONFIG["detect"]


def test_champion_params_parses_block():
    """A config carrying a recall_recovery block is parsed into the tuple."""
    cfg = {
        "detect": {**EMBEDDED_CHAMPION_CONFIG["detect"],
                   "recall_recovery": ["madk_tier", 1.5, 0.2]},
        "link": EMBEDDED_CHAMPION_CONFIG["link"],
    }
    detect, _link, _scale = champion_params(cfg)
    assert detect.recall_recovery == ("madk_tier", 1.5, 0.2)
