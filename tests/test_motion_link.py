"""ARGUS motion-model predicted-position LAP linking (SOT-2864), on synthetic tracks.

Data-free: builds detection dicts by hand and checks (1) that the mechanism is
byte-for-byte off by default, (2) that a global motion field predicts a moving
cell's next position even for a cell with no track history, and (3) that the
predicted-position matching is deterministic — so it runs in CI without the
(gitignored) competition volumes.
"""

from __future__ import annotations

import numpy as np

from biohub_tracking.link import (
    LinkParams,
    _motion_field_predict,
    link_centroids,
)

# isotropic scale keeps the arithmetic transparent in these tests
ISO = (1.0, 1.0, 1.0)


def _uniform_drift():
    # Three cells drifting by a constant +1 in x each frame (a rigid translation).
    base = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    return {t: base + np.array([0.0, 0.0, float(t)]) for t in range(4)}


def test_motion_model_off_is_byte_for_byte_champion():
    dets = _uniform_drift()
    base = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    dflt = link_centroids(
        dets, scale=ISO, params=LinkParams(max_distance=3.0, motion_model_link=False)
    )
    assert base.edges == dflt.edges
    assert base.num_edges == dflt.num_edges


def test_motion_gain_zero_reduces_to_champion_assignment():
    # With gain 0 the predicted position equals the raw position, so the motion
    # path solves the identical assignment as the champion nearest-neighbour path.
    dets = _uniform_drift()
    champ = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    off = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(max_distance=3.0, motion_model_link=True, motion_gain=0.0),
    )
    assert champ.edges == off.edges


def test_motion_field_predicts_global_translation():
    # A rigid +1-in-x translation: the field must recover ~[0,0,1] for every
    # source, including one that the provisional pass would leave to drift.
    src = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    dst = src + np.array([0.0, 0.0, 1.0])
    pred = _motion_field_predict(
        src, dst, np.asarray(ISO, dtype=float), 3.0, smooth_sigma=15.0, gain=1.0
    )
    assert np.allclose(pred - src, [0.0, 0.0, 1.0], atol=0.2)


def test_motion_field_predicts_history_less_cell():
    # A cell appearing for the first time (no prior edge) still inherits its
    # neighbourhood's motion from the global field — the ARGUS distinction from
    # the own-track velocity_gain path.
    src = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    dst = src + np.array([0.0, 0.0, 2.0])
    pred = _motion_field_predict(
        src, dst, np.asarray(ISO, dtype=float), 3.0, smooth_sigma=5.0, gain=1.0
    )
    # every source (all three) gets a non-trivial predicted forward displacement
    disp = pred - src
    assert np.all(disp[:, 2] > 1.0)


def test_motion_linking_is_deterministic():
    dets = _uniform_drift()
    params = LinkParams(
        max_distance=3.0,
        motion_model_link=True,
        motion_smooth_sigma=15.0,
        motion_gain=1.0,
        motion_gate_on_prediction=True,
    )
    a = link_centroids(dets, scale=ISO, params=params)
    b = link_centroids(dets, scale=ISO, params=params)
    assert a.edges == b.edges
