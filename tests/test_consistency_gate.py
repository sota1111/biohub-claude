"""Ultrack bidirectional forward↔backward motion-consistency link gate (SOT-2883).

Data-free (numpy-only synthetic tracks), so it runs in CI without the (gitignored)
competition volumes. Checks that

1. the flag is byte-for-byte off by default and inert without the SOT-2864 motion
   linker (it reuses that field, so it does nothing on the champion path);
2. with the motion linker on, ``tol=inf`` + ``weight=0`` is a strict no-op — the gate
   is a pure superset of the SOT-2864 feasible set (the "accept" path);
3. a finite tol REJECTS a link whose forward/backward field predictions disagree
   (the FP-prone case), while keeping a bidirectionally consistent one;
4. the soft penalty re-ranks toward the bidirectionally consistent successor.
"""

from __future__ import annotations

import numpy as np

from biohub_tracking.link import LinkParams, _assign, link_centroids

ISO = (1.0, 1.0, 1.0)
ISO_ARR = np.asarray(ISO, dtype=float)


def _uniform_drift():
    # Three cells drifting by a constant +1 in x each frame (a rigid translation);
    # forward and backward fields agree everywhere, so every link is consistent.
    base = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    return {t: base + np.array([0.0, 0.0, float(t)]) for t in range(4)}


def _motion(**over):
    kw = dict(max_distance=3.0, motion_model_link=True, motion_gate_on_prediction=True)
    kw.update(over)
    return LinkParams(**kw)


def test_gate_off_is_byte_for_byte_champion():
    dets = _uniform_drift()
    champ = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    off = link_centroids(
        dets, scale=ISO, params=LinkParams(max_distance=3.0, link_consistency_gate=False)
    )
    assert champ.edges == off.edges


def test_gate_inert_without_motion_model():
    # The gate reuses the SOT-2864 field; with motion_model_link off there is no
    # forward/backward prediction to check, so it must be a byte-for-byte no-op.
    dets = _uniform_drift()
    champ = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    on_no_motion = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(
            max_distance=3.0, link_consistency_gate=True, link_consistency_tol=0.1
        ),
    )
    assert champ.edges == on_no_motion.edges


def test_gate_noop_when_tol_inf_and_weight_zero():
    # With the motion linker on, tol=inf + weight=0 is a strict superset no-op.
    dets = _uniform_drift()
    motion = link_centroids(dets, scale=ISO, params=_motion())
    gated = link_centroids(
        dets,
        scale=ISO,
        params=_motion(link_consistency_gate=True),  # tol=inf, weight=0 defaults
    )
    assert motion.edges == gated.edges


def test_gate_keeps_consistent_rigid_translation():
    # On a rigid drift both residuals are ~0, so even a tight tol keeps every
    # SOT-2864 link (the gate never hurts genuinely consistent tracks).
    dets = _uniform_drift()
    motion = link_centroids(dets, scale=ISO, params=_motion())
    gated = link_centroids(
        dets,
        scale=ISO,
        params=_motion(link_consistency_gate=True, link_consistency_tol=0.5),
    )
    assert motion.edges == gated.edges
    assert gated.num_edges > 0


def test_assign_rejects_backward_inconsistent_pair():
    # One feasible pair (forward prediction lands exactly on dst -> r_f=0). When the
    # backward field predicts dst came from far away (r_b large), a finite tol drops
    # it; a consistent backward prediction (r_b=0) keeps it.
    src = np.array([[0.0, 0.0, 0.0]])
    dst = np.array([[0.0, 0.0, 2.0]])
    src_pred = np.array([[0.0, 0.0, 2.0]])  # forward: src -> dst exactly, r_f = 0

    consistent = _assign(
        src, dst, ISO_ARR, 3.0, src_pred=src_pred, gate_on_prediction=True,
        dst_pred_bwd=np.array([[0.0, 0.0, 0.0]]),  # backward -> src exactly, r_b = 0
        consistency_gate=True, consistency_tol=1.0,
    )
    assert consistent == [(0, 0)]

    inconsistent = _assign(
        src, dst, ISO_ARR, 3.0, src_pred=src_pred, gate_on_prediction=True,
        dst_pred_bwd=np.array([[0.0, 0.0, 10.0]]),  # backward disagrees, r_b = 10
        consistency_gate=True, consistency_tol=1.0,
    )
    assert inconsistent == []


def test_assign_soft_penalty_prefers_consistent_successor():
    # Two feasible successors equidistant in raw/forward terms; the backward field
    # agrees with the source only for j=1, so the soft penalty must break the tie
    # toward it (no hard tol; weight only).
    src = np.array([[0.0, 0.0, 0.0]])
    dst = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
    src_pred = np.array([[0.0, 0.0, 0.0]])  # forward neutral: r_f = 2 for both
    # Backward predictions: j=0 returns to src (r_b=0), j=1 returns far (r_b=8).
    dst_pred_bwd = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -8.0]])
    pairs = _assign(
        src, dst, ISO_ARR, 3.0, src_pred=src_pred, gate_on_prediction=False,
        dst_pred_bwd=dst_pred_bwd,
        consistency_gate=True, consistency_weight=1.0,
    )
    assert pairs == [(0, 0)]
