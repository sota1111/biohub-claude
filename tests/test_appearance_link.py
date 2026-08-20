"""Appearance-descriptor-augmented linking (SOT-2829).

Covers three properties of the ``link.appearance_weight`` term:

1. **Byte-invariance** — ``appearance_weight == 0`` (default), and any positive
   weight with ``descriptors=None``, reproduce the distance-only champion graph
   edge-for-edge.
2. **Descriptor extraction** — :func:`patch_descriptors` returns aligned, finite,
   deterministic vectors and handles border/empty inputs.
3. **Disambiguation** — when two successors are equidistant (a tie the distance-only
   linker breaks arbitrarily), a positive appearance weight steers the match to the
   look-alike successor.
"""

from __future__ import annotations

import numpy as np

from biohub_tracking.detect import (
    APPEARANCE_DESCRIPTOR_DIM,
    detect_volume_series_with_descriptors,
    patch_descriptors,
)
from biohub_tracking.link import LinkParams, link_centroids

ISO = (1.0, 1.0, 1.0)


def _edge_set(graph):
    return set(graph.edges)


def test_appearance_weight_zero_is_byte_invariant():
    """Default (weight 0) matches the distance-only champion even with descriptors."""
    dets = {
        0: np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]),
        1: np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 11.0]]),
    }
    descs = {t: np.random.default_rng(t).normal(size=(2, 8)) for t in dets}
    base = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    withd = link_centroids(
        dets, scale=ISO, params=LinkParams(max_distance=3.0), descriptors=descs
    )
    assert _edge_set(withd) == _edge_set(base)


def test_positive_weight_without_descriptors_is_byte_invariant():
    """A positive weight is inert when no descriptors are supplied."""
    dets = {
        0: np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]),
        1: np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 11.0]]),
    }
    base = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    withw = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(max_distance=3.0, appearance_weight=5.0),
        descriptors=None,
    )
    assert _edge_set(withw) == _edge_set(base)


def test_appearance_gate_stays_on_distance():
    """Appearance never admits an out-of-range (metric-invalid) edge."""
    dets = {0: np.array([[0.0, 0.0, 0.0]]), 1: np.array([[0.0, 0.0, 100.0]])}
    descs = {0: np.ones((1, 8)), 1: np.ones((1, 8))}  # perfectly similar
    g = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(max_distance=5.0, appearance_weight=10.0),
        descriptors=descs,
    )
    assert g.num_edges == 0  # too far, however similar-looking


def test_appearance_breaks_equidistant_tie():
    """Two equidistant successors: appearance steers the match to the look-alike."""
    # Source at x=0. Two candidates at x=+2 and x=-2 (both distance 2, feasible).
    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]]),
    }
    # Source looks like candidate index 1 (x=-2), not index 0 (x=+2).
    src_desc = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dst_desc = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # opposite look
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # same look as source
        ]
    )
    descs = {0: src_desc, 1: dst_desc}
    g = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(
            max_distance=3.0, allow_division=False, appearance_weight=2.0
        ),
        descriptors=descs,
    )
    assert g.num_edges == 1
    (_src, dst), = list(g.edges)
    # It must link to the look-alike candidate (x=-2), not the other equidistant one.
    assert g.position(dst)[2] == -2.0


def test_patch_descriptors_shape_and_finite():
    rng = np.random.default_rng(0)
    vol = rng.random((8, 16, 16)).astype(np.float32)
    coords = np.array([[4.0, 8.0, 8.0], [0.0, 0.0, 0.0], [7.0, 15.0, 15.0]])
    desc = patch_descriptors(vol, coords)
    assert desc.shape == (3, APPEARANCE_DESCRIPTOR_DIM)
    assert np.isfinite(desc).all()
    # deterministic
    desc2 = patch_descriptors(vol, coords)
    assert np.array_equal(desc, desc2)


def test_patch_descriptors_empty():
    desc = patch_descriptors(np.zeros((4, 4, 4), dtype=np.float32), np.zeros((0, 3)))
    assert desc.shape == (0, APPEARANCE_DESCRIPTOR_DIM)


def test_detect_series_with_descriptors_aligns_and_standardises():
    rng = np.random.default_rng(1)
    # (T, Z, Y, X) with a couple of bright blobs so the detector fires.
    vol = rng.random((3, 8, 24, 24)).astype(np.float32) * 0.1
    vol[:, 4, 6:9, 6:9] = 5.0
    vol[:, 4, 16:19, 16:19] = 5.0
    dets, descs = detect_volume_series_with_descriptors(vol)
    assert set(dets) == set(descs)
    for t in dets:
        assert len(dets[t]) == len(descs[t])
        assert descs[t].shape[1] == APPEARANCE_DESCRIPTOR_DIM
    # Standardised: each feature dimension ~ zero-mean across the whole video.
    stacked = np.concatenate([d for d in descs.values() if len(d)], axis=0)
    assert np.allclose(stacked.mean(axis=0), 0.0, atol=1e-6)
