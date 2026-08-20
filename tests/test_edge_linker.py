"""GT-learned edge-linking cost (SOT-2841).

Covers the four properties the issue's acceptance criteria pin:

1. **Byte-invariance** — ``edge_cost_model is None`` (default), a model with
   ``weight == 0``, and any model with ``descriptors=None``, all reproduce the
   distance-only champion graph edge-for-edge.
2. **Metric-validity** — the learned term never admits an out-of-range edge (the
   ``<= max_distance`` gate stays on the raw scaled distance).
3. **Feature/inference plumbing** — :func:`edge_feature_planes` is deterministic,
   correctly shaped, and its ranks/margins are computed over the feasible set; the
   embedded logistic round-trips through ``to_dict``/``from_dict``.
4. **Disambiguation** — a model that has *learned* to prefer the look-alike
   successor steers an equidistant tie to it (the dense-family failure mode).
"""

from __future__ import annotations

import numpy as np

from biohub_tracking.edge_linker import (
    EDGE_FEATURE_NAMES,
    N_EDGE_FEATURES,
    LearnedEdgeCost,
    edge_feature_planes,
    fit_edge_cost,
)
from biohub_tracking.link import LinkParams, link_centroids

ISO = (1.0, 1.0, 1.0)


def _edge_set(graph):
    return set(graph.edges)


def _identity_model(weight: float) -> dict:
    """A zeroed-coef model (p_edge == 0.5 everywhere) at a given weight."""
    return LearnedEdgeCost(
        feature_names=EDGE_FEATURE_NAMES,
        mean=np.zeros(N_EDGE_FEATURES),
        std=np.ones(N_EDGE_FEATURES),
        coef=np.zeros(N_EDGE_FEATURES),
        intercept=0.0,
        weight=weight,
    ).to_dict()


def test_default_none_is_byte_invariant():
    """No edge model (default) == distance-only champion, even with descriptors."""
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


def test_weight_zero_model_is_byte_invariant():
    """A model with weight 0 drops the term (byte-for-byte champion)."""
    dets = {
        0: np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]),
        1: np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 11.0]]),
    }
    descs = {t: np.random.default_rng(t).normal(size=(2, 8)) for t in dets}
    base = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    withm = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(max_distance=3.0, edge_cost_model=_identity_model(0.0)),
        descriptors=descs,
    )
    assert _edge_set(withm) == _edge_set(base)


def test_positive_weight_without_descriptors_is_byte_invariant():
    """A positive-weight model is inert when no descriptors are supplied."""
    dets = {
        0: np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]),
        1: np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 11.0]]),
    }
    base = link_centroids(dets, scale=ISO, params=LinkParams(max_distance=3.0))
    withm = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(max_distance=3.0, edge_cost_model=_identity_model(5.0)),
        descriptors=None,
    )
    assert _edge_set(withm) == _edge_set(base)


def test_gate_stays_on_distance():
    """The learned term never admits an out-of-range (metric-invalid) edge."""
    dets = {0: np.array([[0.0, 0.0, 0.0]]), 1: np.array([[0.0, 0.0, 100.0]])}
    descs = {0: np.ones((1, 8)), 1: np.ones((1, 8))}
    # A model that would love this pair (large positive intercept -> p_edge ~ 1).
    model = LearnedEdgeCost(
        feature_names=EDGE_FEATURE_NAMES,
        mean=np.zeros(N_EDGE_FEATURES),
        std=np.ones(N_EDGE_FEATURES),
        coef=np.zeros(N_EDGE_FEATURES),
        intercept=50.0,
        weight=10.0,
    ).to_dict()
    g = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(max_distance=5.0, edge_cost_model=model),
        descriptors=descs,
    )
    assert g.num_edges == 0  # too far, however much the model likes it


def test_learned_model_breaks_equidistant_tie():
    """A model trained to prefer the look-alike steers an equidistant tie to it."""
    # Two candidates at x=+2 and x=-2 (both feasible, equidistant). Build a tiny
    # training set where the GT successor is always the appearance look-alike, so the
    # fitted model puts positive weight on app_cos.
    rng = np.random.default_rng(0)
    feats, labels = [], []
    for _ in range(200):
        # positive: high app_cos, rank 0; negative: low app_cos, rank 1.
        # columns: dist, app_cos, src_rivals, dst_rivals, succ_rank, pred_rank, margin
        feats.append([2.0, 0.95 + 0.02 * rng.standard_normal(), 1, 0, 0, 0, 0.0])
        labels.append(1.0)
        feats.append([2.0, 0.10 + 0.02 * rng.standard_normal(), 1, 0, 1, 0, 0.0])
        labels.append(0.0)
    model = fit_edge_cost(
        np.array(feats), np.array(labels), weight=3.0, iters=2000, lr=0.5
    )

    dets = {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]]),
    }
    # Source looks like candidate index 1 (x=-2).
    src_desc = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dst_desc = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # opposite look
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # look-alike
        ]
    )
    descs = {0: src_desc, 1: dst_desc}
    g = link_centroids(
        dets,
        scale=ISO,
        params=LinkParams(
            max_distance=3.0, allow_division=False, edge_cost_model=model.to_dict()
        ),
        descriptors=descs,
    )
    assert g.num_edges == 1
    (_src, dst), = list(g.edges)
    assert g.position(dst)[2] == -2.0  # linked to the look-alike, not the other


def test_feature_planes_shape_and_determinism():
    dist = np.array([[1.0, 2.0, 20.0], [3.0, 0.5, 4.0]])
    src_desc = np.random.default_rng(1).normal(size=(2, 8))
    dst_desc = np.random.default_rng(2).normal(size=(3, 8))
    planes = edge_feature_planes(dist, src_desc, dst_desc, max_distance=5.0)
    assert planes.shape == (2, 3, N_EDGE_FEATURES)
    assert np.isfinite(planes).all()
    planes2 = edge_feature_planes(dist, src_desc, dst_desc, max_distance=5.0)
    assert np.array_equal(planes, planes2)
    # dist_scaled feature == the input distance matrix.
    assert np.array_equal(planes[:, :, 0], dist)
    # succ_rank: row 0 has feasibles {1.0 (col0), 2.0 (col1)}; col2 (20>5) infeasible.
    assert planes[0, 0, 4] == 0.0  # nearest feasible successor of src 0
    assert planes[0, 1, 4] == 1.0  # second nearest
    # succ_margin over the source's nearest feasible successor.
    assert planes[0, 0, 6] == 0.0
    assert planes[0, 1, 6] == 1.0  # 2.0 - 1.0


def test_model_dict_roundtrip():
    model = LearnedEdgeCost(
        feature_names=EDGE_FEATURE_NAMES,
        mean=np.arange(N_EDGE_FEATURES, dtype=float),
        std=np.ones(N_EDGE_FEATURES) * 2.0,
        coef=np.linspace(-1, 1, N_EDGE_FEATURES),
        intercept=0.3,
        weight=1.5,
    )
    back = LearnedEdgeCost.from_dict(model.to_dict())
    assert back.feature_names == EDGE_FEATURE_NAMES
    assert np.allclose(back.mean, model.mean)
    assert np.allclose(back.coef, model.coef)
    assert back.intercept == 0.3
    assert back.weight == 1.5


def test_from_dict_rejects_feature_drift():
    d = _identity_model(1.0)
    d["feature_names"] = list(EDGE_FEATURE_NAMES)[:-1] + ["bogus"]
    try:
        LearnedEdgeCost.from_dict(d)
    except ValueError:
        return
    raise AssertionError("expected ValueError on feature_names drift")
