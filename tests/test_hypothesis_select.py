"""Unit tests for Ultrack multi-hypothesis detection selection (SOT-2884).

The knob keeps a threshold-ladder candidate *pool* per frame and selects the final
disjoint detections by **temporal support** (only candidates that link into a track
of length >= L survive). It is a series-level detection re-anchor wired in
:func:`biohub_tracking.pipeline.run_pipeline`; it must be byte-for-byte off by
default, the per-frame champion detector must be untouched, the pool must be a
superset of the champion peaks, and temporal support must drop persistence-less
sub-threshold candidates while keeping supported ones.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from biohub_tracking.champion import EMBEDDED_CHAMPION_CONFIG, champion_params
from biohub_tracking.detect import (
    DetectParams,
    candidate_pool,
    detect_centroids,
    detect_centroids_with_meta,
    detect_volume_series,
)
from biohub_tracking.link import LinkParams, link_centroids
from biohub_tracking.pipeline import _hypothesis_select_detections


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


# --- per-frame detector is untouched (default-off byte-invariance) -----------

def test_flag_off_leaves_per_frame_detector_byte_identical():
    """detect_hypothesis_select is a series-level knob; the per-volume detector is
    byte-for-byte the champion whether the flag is set or not (it is never read
    inside detect_centroids)."""
    vol = _blob_volume([(8, 16, 16), (8, 48, 48), (4, 32, 20)], amps=[600.0, 400.0, 250.0])
    base = detect_centroids(vol, _CHAMP)
    off = detect_centroids(vol, _with(detect_hypothesis_select=False))
    on = detect_centroids(vol, _with(detect_hypothesis_select=True, hypothesis_mad_k_low=2.0))
    assert np.array_equal(base, off)
    assert np.array_equal(base, on)


# --- candidate pool is a superset of the champion peaks ----------------------

def test_pool_equals_champion_peaks_when_low_gate_equals_madk():
    """With hypothesis_mad_k_low == mad_k the pool is exactly the champion primary
    peaks (same response, same NMS, same gate) — a clean superset baseline."""
    vol = _blob_volume([(8, 16, 16), (8, 48, 48), (4, 32, 20)], amps=[600.0, 400.0, 250.0])
    champ, _ = detect_centroids_with_meta(vol, _CHAMP)
    pool, strengths = candidate_pool(vol, _with(hypothesis_mad_k_low=3.0))
    assert np.array_equal(np.sort(pool, axis=0), np.sort(champ, axis=0))
    # brightest-first ordering
    assert np.all(np.diff(strengths) <= 1e-9)


def test_pool_is_superset_with_lower_gate():
    """A lower gate admits at least as many candidates as the champion cutoff, and
    every champion peak is still in the pool."""
    rng = np.random.default_rng(3)
    centers = [(8, 12, 20), (8, 40, 40), (4, 24, 55)]
    vol = _blob_volume(centers, amps=[900.0, 850.0, 800.0], shape=(16, 80, 80))
    vol = (vol + rng.normal(0.0, 20.0, size=vol.shape).astype(np.float32)).astype(np.float32)
    champ, _ = detect_centroids_with_meta(vol, _CHAMP)
    pool, _ = candidate_pool(vol, _with(hypothesis_mad_k_low=1.5))
    assert pool.shape[0] >= champ.shape[0]
    champ_set = {tuple(np.rint(c)) for c in champ}
    pool_set = {tuple(np.rint(c)) for c in pool}
    assert champ_set <= pool_set


def test_pool_cap_bounds_size():
    """hypothesis_pool_cap keeps only the strongest candidates."""
    rng = np.random.default_rng(0)
    centers = [(8, 6 + 5 * i, 6 + 4 * (i % 12)) for i in range(30)]
    amps = list(rng.uniform(200.0, 900.0, size=30))
    vol = _blob_volume(centers, amps=amps, shape=(16, 72, 72))
    pool_uncapped, _ = candidate_pool(vol, _with(hypothesis_mad_k_low=0.5))
    pool_capped, s = candidate_pool(
        vol, _with(hypothesis_mad_k_low=0.5, hypothesis_pool_cap=5)
    )
    assert pool_capped.shape[0] == min(5, pool_uncapped.shape[0])
    assert np.all(np.diff(s) <= 1e-9)  # brightest-first retained under the cap


def test_pool_requires_mad_k():
    """The pool needs the adaptive robust threshold path."""
    vol = _blob_volume([(8, 16, 16)], amps=[600.0])
    with pytest.raises(ValueError, match="mad_k"):
        candidate_pool(vol, _with(mad_k=None))


# --- temporal support selection (series level) -------------------------------

def _moving_series(n_t=6, shape=(12, 64, 64)):
    """A persistent cell that drifts smoothly frame-to-frame plus a per-frame
    ephemeral blob that *teleports* > max_distance each frame. The persistent cell
    links into a length-n_t track; each ephemeral blob is an isolated singleton
    (consecutive-frame distance exceeds the 7 µm gate, so it never links)."""
    vols = []
    for t in range(n_t):
        persistent = (6, 12 + t * 2, 12 + t * 2)  # ~1.15 µm/frame drift (in-gate)
        # Far corner, x alternates 33<->55 voxels: 22 * 0.40625 ≈ 8.9 µm/frame > 7 µm
        # gate (never self-links) and > 13 µm from the persistent drift path.
        ephemeral = (6, 55, 33 + 22 * (t % 2))
        vol = _blob_volume([persistent, ephemeral], amps=[800.0, 780.0], shape=shape)
        vols.append(vol)
    return np.stack(vols, axis=0)


def test_temporal_support_keeps_persistent_drops_ephemeral():
    """With L=n_t the only surviving track is the persistent drifting cell; the
    ephemeral per-frame blobs (no temporal support) are dropped, so the linked graph
    is a single long track."""
    arr = _moving_series(n_t=6)
    scale = (1.625, 0.40625, 0.40625)
    link = LinkParams(max_distance=7.0, min_track_length=1)
    detect = _with(detect_hypothesis_select=True, hypothesis_mad_k_low=2.0, hypothesis_min_track=6)
    detections = _hypothesis_select_detections(arr, detect, link, scale, None)
    graph = link_centroids(detections, scale=scale, params=link)
    # Exactly the 6-node persistent track survives (one node per timepoint).
    assert graph.num_nodes == 6
    times = sorted(graph.t(n) for n in graph.node_ids())
    assert times == list(range(6))


def test_min_track_one_keeps_whole_pool():
    """L=1 imposes no temporal filter: every pooled candidate survives (both the
    persistent and the ephemeral blob in each frame)."""
    arr = _moving_series(n_t=4)
    scale = (1.625, 0.40625, 0.40625)
    link = LinkParams(max_distance=7.0, min_track_length=1)
    detect = _with(detect_hypothesis_select=True, hypothesis_mad_k_low=2.0, hypothesis_min_track=1)
    detections = _hypothesis_select_detections(arr, detect, link, scale, None)
    total = sum(len(v) for v in detections.values())
    assert total == 8  # 2 blobs * 4 frames


def test_flag_off_pool_matches_classical_detect_series():
    """The pool at L=1, low-gate==mad_k reproduces the classical per-frame detections
    (the selection layer only *removes* — with L=1 nothing is removed and the pool
    at the champion gate is exactly the champion peaks)."""
    arr = _moving_series(n_t=4)
    scale = (1.625, 0.40625, 0.40625)
    link = LinkParams(max_distance=7.0, min_track_length=1)
    classical = detect_volume_series(arr, _CHAMP)
    detect = _with(detect_hypothesis_select=True, hypothesis_mad_k_low=3.0, hypothesis_min_track=1)
    selected = _hypothesis_select_detections(arr, detect, link, scale, None)
    for t in classical:
        a = np.asarray(sorted(map(tuple, classical[t])))
        b = np.asarray(sorted(map(tuple, selected[t])))
        assert np.array_equal(a, b)


# --- champion config plumbing ------------------------------------------------

def test_champion_params_absent_key_is_off():
    """The embedded champion config carries no hypothesis-select keys → flag off."""
    detect, _link, _scale = champion_params()
    assert detect.detect_hypothesis_select is False
    assert detect.hypothesis_mad_k_low is None
    assert detect.hypothesis_min_track == 3
    assert "detect_hypothesis_select" not in EMBEDDED_CHAMPION_CONFIG["detect"]


def test_champion_params_parses_block():
    """A config carrying the hypothesis-select block is parsed onto DetectParams."""
    cfg = {
        "detect": {
            **EMBEDDED_CHAMPION_CONFIG["detect"],
            "detect_hypothesis_select": True,
            "hypothesis_mad_k_low": 2.0,
            "hypothesis_min_track": 4,
            "hypothesis_pool_cap": 12000,
        },
        "link": EMBEDDED_CHAMPION_CONFIG["link"],
    }
    detect, _link, _scale = champion_params(cfg)
    assert detect.detect_hypothesis_select is True
    assert detect.hypothesis_mad_k_low == 2.0
    assert detect.hypothesis_min_track == 4
    assert detect.hypothesis_pool_cap == 12000
