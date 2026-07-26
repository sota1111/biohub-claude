"""Centroid-distance node matching.

The competition matches predicted nodes to ground-truth nodes *per timepoint* by
their 3D centroids, using an **optimal (minimum-cost) bipartite assignment** with
a maximum distance of **7 µm**. Matching is one-to-one: each predicted node pairs
with at most one ground-truth node and vice versa. Distances are computed in
physical (micron) space, so the anisotropic voxel ``scale`` matters.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .graph import TrackingGraph

# Physical voxel size (z, y, x) in microns for the biohub datasets. Read from the
# geff metadata when available; this is the documented default.
DEFAULT_SCALE: tuple[float, float, float] = (1.625, 0.40625, 0.40625)

# Maximum centroid distance (microns) for a node match.
DEFAULT_MAX_DISTANCE: float = 7.0


def match_nodes(
    pred: TrackingGraph,
    gt: TrackingGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    gt_node_subset: Iterable[int] | None = None,
) -> dict[int, int]:
    """Return a ``pred_id -> gt_id`` mapping of matched nodes.

    Matching is solved independently for each timepoint that appears in both
    graphs. Within a timepoint the scaled Euclidean centroid distances form a
    cost matrix; :func:`scipy.optimize.linear_sum_assignment` finds the
    minimum-cost one-to-one assignment, and any pair farther apart than
    *max_distance* is discarded.

    Parameters
    ----------
    pred, gt
        Predicted and ground-truth graphs.
    scale
        Physical ``(z, y, x)`` voxel scale. ``None`` means isotropic (all ones),
        matching the reference behaviour when no scale is supplied.
    max_distance
        Maximum scaled centroid distance for a valid match.
    gt_node_subset
        Restrict matching to this subset of GT node ids (used by the division
        metric, which matches against a small window of GT nodes at a time).
    """
    scale_arr = np.ones(3, dtype=float) if scale is None else np.asarray(scale, dtype=float)

    gt_ids_allowed = None if gt_node_subset is None else set(int(n) for n in gt_node_subset)

    pred_by_t = pred.nodes_by_time()
    gt_by_t = gt.nodes_by_time()

    matching: dict[int, int] = {}
    for t, gt_ids in gt_by_t.items():
        if gt_ids_allowed is not None:
            gt_ids = [g for g in gt_ids if g in gt_ids_allowed]
        pred_ids = pred_by_t.get(t, [])
        if not gt_ids or not pred_ids:
            continue

        pred_pos = np.stack([pred.position(p) for p in pred_ids])  # (P, 3)
        gt_pos = np.stack([gt.position(g) for g in gt_ids])  # (G, 3)
        # Scaled Euclidean distance matrix (P, G) in microns.
        diff = (pred_pos[:, None, :] - gt_pos[None, :, :]) * scale_arr[None, None, :]
        cost = np.sqrt((diff**2).sum(axis=2))

        # Mask pairs beyond the threshold with a prohibitive finite cost so the
        # assignment never prefers an out-of-range pair over a valid one when the
        # totals would otherwise tie. Masked pairs are dropped after assignment.
        big = float(cost.max()) * cost.size + 1.0
        masked = np.where(cost > max_distance, big, cost)

        rows, cols = linear_sum_assignment(masked)
        for r, c in zip(rows, cols):
            if cost[r, c] <= max_distance:
                matching[pred_ids[r]] = gt_ids[c]

    return matching
