"""Leak-free / exec-compat tests for the learned cross-attention edge cost (SOT-2994).

Guards the invariants the promotion contract relies on:

* **Default-off byte-invariance**: ``weight == 0`` (or no model / no descriptors) makes
  the penalty identically zero and ``link_centroids`` reproduces the champion graph
  edge-for-edge — the frozen champion config stays byte-identical.
* **Pure-numpy / GT-free inference**: the forward pass is a function of the distance
  matrix + descriptors only (no ground truth, no torch, no filesystem) and is
  deterministic under input shuffling of unrelated rows.
* **Serialization round-trip + guards**: ``to_dict``/``from_dict`` is exact and rejects
  a drifted feature order or architecture.
* **Masked attention**: infeasible pairs receive exactly zero attention weight, so an
  out-of-gate candidate can never leak into a context vector.
* **Train/infer parity** (torch-gated): the shipped numpy forward matches a torch
  re-implementation of the same forward on the trained weights — no skew.
"""

from __future__ import annotations

import numpy as np
import pytest

from biohub_tracking.edge_linker import N_EDGE_FEATURES, edge_feature_planes
from biohub_tracking.link import LinkParams, link_centroids
from biohub_tracking.xattn_edge import (
    XATTN_ARCH_VERSION,
    XATTN_FEATURE_NAMES,
    CrossAttentionEdgeCost,
    torch_available,
)

RNG = np.random.default_rng(0)


def _random_model(weight: float, d: int = 6, h2: int = 5) -> CrossAttentionEdgeCost:
    F = N_EDGE_FEATURES
    return CrossAttentionEdgeCost(
        feature_names=XATTN_FEATURE_NAMES,
        d=d,
        h2=h2,
        mean=RNG.normal(size=F),
        std=np.abs(RNG.normal(size=F)) + 0.5,
        W1=RNG.normal(size=(F, d)) * 0.3,
        b1=RNG.normal(size=d) * 0.1,
        a_row=RNG.normal(size=d) * 0.3,
        a_col=RNG.normal(size=d) * 0.3,
        W2=RNG.normal(size=(3 * d, h2)) * 0.2,
        b2=RNG.normal(size=h2) * 0.1,
        w3=RNG.normal(size=h2) * 0.3,
        b3=float(RNG.normal()),
        weight=weight,
    )


def _toy_frame(n_src=5, n_dst=6):
    src = RNG.uniform(0, 20, size=(n_src, 3))
    dst = RNG.uniform(0, 20, size=(n_dst, 3))
    scale = np.array([1.625, 0.40625, 0.40625])
    diff = (src[:, None, :] - dst[None, :, :]) * scale
    dist = np.sqrt((diff**2).sum(axis=2))
    src_desc = RNG.normal(size=(n_src, 8))
    dst_desc = RNG.normal(size=(n_dst, 8))
    return dist, src_desc, dst_desc


def test_weight_zero_penalty_is_identically_zero():
    model = _random_model(weight=0.0)
    dist, sd, dd = _toy_frame()
    pen = model.penalty(dist, sd, dd, max_distance=7.0)
    assert pen.shape == dist.shape
    assert np.all(pen == 0.0)


def test_probability_is_gt_free_and_deterministic():
    """Inference reads only distance + descriptors — no labels — and is deterministic."""
    model = _random_model(weight=2.0)
    dist, sd, dd = _toy_frame()
    p1 = model.probability_planes(dist, sd, dd, max_distance=7.0)
    p2 = model.probability_planes(dist.copy(), sd.copy(), dd.copy(), max_distance=7.0)
    assert np.array_equal(p1, p2)
    assert p1.shape == dist.shape
    assert np.all((p1 >= 0.0) & (p1 <= 1.0))


def test_to_from_dict_roundtrip_exact():
    model = _random_model(weight=1.5)
    back = CrossAttentionEdgeCost.from_dict(model.to_dict())
    dist, sd, dd = _toy_frame()
    assert np.allclose(
        model.probability_planes(dist, sd, dd, 7.0),
        back.probability_planes(dist, sd, dd, 7.0),
        atol=1e-12,
    )
    assert back.weight == model.weight
    assert back.arch == XATTN_ARCH_VERSION


def test_from_dict_rejects_feature_drift():
    d = _random_model(weight=1.0).to_dict()
    d["feature_names"] = list(d["feature_names"])[:-1] + ["bogus"]
    with pytest.raises(ValueError):
        CrossAttentionEdgeCost.from_dict(d)


def test_from_dict_rejects_arch_drift():
    d = _random_model(weight=1.0).to_dict()
    d["arch"] = "some-other-arch-v9"
    with pytest.raises(ValueError):
        CrossAttentionEdgeCost.from_dict(d)


def test_masked_attention_ignores_infeasible_pairs():
    """An infeasible column contributes zero attention: at the forward level (fixed
    feasibility mask, fixed planes for the feasible pairs) perturbing an infeasible
    column's feature planes must not change any feasible pair's p_edge."""
    model = _random_model(weight=2.0)
    dist, sd, dd = _toy_frame(n_src=4, n_dst=5)
    planes = edge_feature_planes(dist, sd, dd, 7.0)
    feasible = np.ones(dist.shape, dtype=bool)
    feasible[:, 0] = False  # column 0 infeasible for every source
    p_before = model.score_transition(planes, feasible)
    planes_perturbed = planes.copy()
    planes_perturbed[:, 0, :] += RNG.normal(size=(dist.shape[0], N_EDGE_FEATURES)) * 5.0
    p_after = model.score_transition(planes_perturbed, feasible)
    # Feasible pairs (columns 1..) are unchanged; only the infeasible column moved.
    assert np.allclose(p_before[:, 1:], p_after[:, 1:], atol=1e-12)


def test_score_transition_matches_probability_planes():
    model = _random_model(weight=1.0)
    dist, sd, dd = _toy_frame()
    planes = edge_feature_planes(dist, sd, dd, 7.0)
    feasible = dist <= 7.0
    assert np.allclose(
        model.score_transition(planes, feasible),
        model.probability_planes(dist, sd, dd, 7.0),
        atol=1e-12,
    )


def test_link_centroids_weight_zero_is_champion_byte_for_byte():
    """A champion-shaped link with an xattn model at weight 0 == champion edges."""
    detections = {
        0: RNG.uniform(0, 20, size=(6, 3)),
        1: RNG.uniform(0, 20, size=(6, 3)),
        2: RNG.uniform(0, 20, size=(6, 3)),
    }
    descriptors = {t: RNG.normal(size=(6, 8)) for t in detections}
    scale = (1.625, 0.40625, 0.40625)
    champ = LinkParams(max_distance=7.0, motion_model_link=True,
                       motion_gate_on_prediction=True)
    g_champ = link_centroids(detections, scale=scale, params=champ)
    model0 = _random_model(weight=0.0)
    with_model = LinkParams(max_distance=7.0, motion_model_link=True,
                            motion_gate_on_prediction=True,
                            xattn_edge_model=model0.to_dict())
    g_with = link_centroids(detections, scale=scale, params=with_model,
                            descriptors=descriptors)
    assert g_champ.edges == g_with.edges
    # And absent descriptors, an active-weight model is inert too (no desc → off).
    g_nodesc = link_centroids(
        detections, scale=scale,
        params=LinkParams(max_distance=7.0, motion_model_link=True,
                          motion_gate_on_prediction=True,
                          xattn_edge_model=_random_model(weight=3.0).to_dict()),
    )
    assert g_champ.edges == g_nodesc.edges


def test_numpy_torch_forward_parity():
    """The shipped numpy forward matches a torch re-implementation on trained weights."""
    if not torch_available():
        pytest.skip("torch not importable; numpy inference path is the shipped one")
    import torch

    from biohub_tracking.xattn_edge import fit_cross_attention

    # Two tiny synthetic transitions with a trainable structure.
    transitions = []
    for _ in range(3):
        dist, sd, dd = _toy_frame(n_src=5, n_dst=6)
        planes = edge_feature_planes(dist, sd, dd, 7.0)
        feasible = dist <= 7.0
        trainable = feasible.copy()
        labels = (RNG.uniform(size=dist.shape) > 0.7).astype(float) * feasible
        transitions.append(
            {"planes": planes, "feasible": feasible,
             "trainable": trainable, "labels": labels}
        )
    model = fit_cross_attention(transitions, weight=1.0, d=6, h2=5, epochs=5, seed=1)

    dist, sd, dd = _toy_frame(n_src=4, n_dst=7)
    planes = edge_feature_planes(dist, sd, dd, 7.0)
    feasible = dist <= 7.0
    p_np = model.score_transition(planes, feasible)

    # Torch re-implementation of the same forward on the model's numpy weights.
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)
    z = (t(planes) - t(model.mean)) / t(np.where(model.std > 1e-12, model.std, 1.0))
    h = torch.relu(z @ t(model.W1) + t(model.b1))
    feas = t(feasible.astype(float)).bool()
    ninf = torch.tensor(-1e30, dtype=torch.float64)
    er = torch.where(feas, h @ t(model.a_row), ninf)
    ar = torch.where(feas, torch.softmax(er, dim=1), torch.zeros_like(er))
    cr = torch.einsum("pq,pqd->pd", ar, h).unsqueeze(1).expand(-1, h.shape[1], -1)
    ec = torch.where(feas, h @ t(model.a_col), ninf)
    ac = torch.where(feas, torch.softmax(ec, dim=0), torch.zeros_like(ec))
    cc = torch.einsum("pq,pqd->qd", ac, h).unsqueeze(0).expand(h.shape[0], -1, -1)
    gg = torch.cat([h, cr, cc], dim=2)
    logit = torch.relu(gg @ t(model.W2) + t(model.b2)) @ t(model.w3) + t(model.b3)
    p_torch = torch.sigmoid(logit).detach().numpy()

    assert np.allclose(p_np, p_torch, atol=1e-9)
