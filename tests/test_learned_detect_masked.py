"""SOT-2993 — masked sparse supervision unit tests.

The acceptance-critical property of the self-trained 3D U-Net detector is that its
loss is MASKED to the annotated field of view: an unannotated cell far from any GT
annotation must be EXCLUDED from the loss (weight exactly 0), so it is never taught
as background (the Positive-Unlabeled contamination cure vs SOT-2828/2848/2863).

These tests pin that property in pure numpy (so they run in the torch-free CI env)
and, when torch is available, prove the gradient really is zero outside the mask.
"""

from __future__ import annotations

import numpy as np
import pytest

from biohub_tracking.learned_detect import (
    annotation_supervision_mask,
    gaussian_heatmap_target,
    masked_sparse_loss_weights,
    torch_available,
)

SHAPE = (8, 48, 48)
SIGMA = (1.0, 3.0, 3.0)
RADIUS = (4.0, 12.0, 12.0)


def test_gaussian_target_peaks_at_points():
    pts = np.array([[4, 12, 12], [4, 36, 36]], dtype=np.float64)
    target = gaussian_heatmap_target(SHAPE, pts, SIGMA)
    assert target.shape == SHAPE
    assert target.dtype == np.float32
    # Peak ~1.0 at each annotated centre, ~0 far away.
    assert target[4, 12, 12] == pytest.approx(1.0, abs=1e-4)
    assert target[4, 36, 36] == pytest.approx(1.0, abs=1e-4)
    assert target[4, 24, 24] < 1e-3  # midway between blobs is background


def test_empty_points_yield_zero_target_and_empty_mask():
    empty = np.zeros((0, 3), dtype=np.float64)
    assert not gaussian_heatmap_target(SHAPE, empty, SIGMA).any()
    assert not annotation_supervision_mask(SHAPE, empty, RADIUS).any()
    # No annotation ⇒ nothing supervised ⇒ all weights zero.
    assert not masked_sparse_loss_weights(SHAPE, empty).any()


def test_mask_is_bounded_ellipsoid_around_annotations():
    pts = np.array([[4, 12, 12]], dtype=np.float64)
    mask = annotation_supervision_mask(SHAPE, pts, RADIUS)
    assert mask.dtype == bool
    # Inside the ellipsoid: supervised.
    assert mask[4, 12, 12]
    assert mask[4, 12, 12 + 6]  # dx=6 < rx=12
    # Outside the ellipsoid: NOT supervised (this is the sparse mask, not all-ones).
    assert not mask[4, 12, 12 + 20]  # dx=20 > rx=12
    assert not mask[4, 40, 40]
    # The mask must NOT cover the whole volume (else it degenerates to naive loss).
    assert mask.mean() < 0.5


def test_unannotated_cell_is_excluded_from_loss():
    """The core SOT-2993 guarantee: a real cell FAR from any GT annotation gets
    weight exactly 0 (excluded from backprop), while the annotated cell and its
    local background are supervised."""
    gt = np.array([[4, 12, 12]], dtype=np.float64)  # the single annotated cell
    w = masked_sparse_loss_weights(SHAPE, gt, sigma_zyx=SIGMA, radius_zyx=RADIUS, fg_weight=50.0)

    # Annotated blob core → foreground weight.
    assert w[4, 12, 12] == pytest.approx(50.0)
    # Local background inside the supervised FOV → weight 1.0.
    assert w[4, 12, 12 + 8] == pytest.approx(1.0)  # dx=8 < rx=12, target<pos_core
    # An UNANNOTATED cell far away (mask False) → weight 0 (EXCLUDED from loss).
    unannotated_cell = (4, 40, 40)
    assert w[unannotated_cell] == 0.0
    # Every foreground-weighted voxel must lie inside the supervised mask.
    mask = annotation_supervision_mask(SHAPE, gt, RADIUS)
    assert np.all(mask[w > 0])


def test_all_blob_cores_are_inside_the_mask():
    """Positive supervision (blob cores) must never be masked out, whatever the
    radius/sigma pairing shipped — otherwise the detector gets no positive signal."""
    pts = np.array([[4, 12, 12], [2, 30, 8]], dtype=np.float64)
    target = gaussian_heatmap_target(SHAPE, pts, SIGMA)
    mask = annotation_supervision_mask(SHAPE, pts, RADIUS)
    cores = target >= 0.5
    assert cores.any()
    assert np.all(mask[cores]), "blob cores must be supervised (inside the mask)"


@pytest.mark.skipif(not torch_available(), reason="torch not importable (classical/CI env)")
def test_masked_loss_gradient_zero_outside_mask():
    """With torch: a per-voxel MSE weighted by the masked weights produces ZERO
    gradient w.r.t. logits outside the mask, and non-zero inside."""
    import torch

    gt = np.array([[4, 12, 12]], dtype=np.float64)
    target = torch.from_numpy(gaussian_heatmap_target(SHAPE, gt, SIGMA))
    weights = torch.from_numpy(masked_sparse_loss_weights(SHAPE, gt, sigma_zyx=SIGMA, radius_zyx=RADIUS))
    logits = torch.zeros(SHAPE, requires_grad=True)

    prob = torch.sigmoid(logits)
    loss = (weights * (prob - target) ** 2).sum() / weights.sum().clamp(min=1.0)
    loss.backward()

    grad = logits.grad.detach().numpy()
    mask = annotation_supervision_mask(SHAPE, gt, RADIUS)
    # Excluded region: exactly zero gradient (never pushed toward background).
    assert np.allclose(grad[~mask], 0.0)
    # Supervised region: real gradient exists.
    assert np.abs(grad[mask]).sum() > 0.0
