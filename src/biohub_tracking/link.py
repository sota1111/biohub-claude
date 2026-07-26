"""Nearest-neighbour frame-to-frame linking with division handling.

Given per-timepoint centroids (from :mod:`biohub_tracking.detect`), build a
:class:`~biohub_tracking.graph.TrackingGraph`:

* **Assignment** — between consecutive timepoints ``t`` and ``t+1`` the scaled
  centroid distances form a cost matrix, solved with an optimal (minimum-cost)
  bipartite assignment (:func:`scipy.optimize.linear_sum_assignment`). Any pair
  farther apart than ``max_distance`` microns is rejected, so a cell that has no
  plausible successor simply ends its track (and a new detection starts one).
* **Division** — after the one-to-one assignment, each still-unmatched ``t+1``
  detection that lies within ``division_distance`` microns of an already-matched
  ``t`` parent is attached as that parent's *second* child, encoding a ``1 -> 2``
  split. A parent gains at most one extra child, so out-degree is capped at two,
  matching the competition's division definition.

The routine is deterministic and mirrors the competition's own micron-space,
optimal-assignment matching, so offline scores track the leaderboard metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .graph import TrackingGraph
from .matching import DEFAULT_MAX_DISTANCE, DEFAULT_SCALE


@dataclass(frozen=True)
class LinkParams:
    """Tunable knobs for :func:`link_centroids` (distances in microns)."""

    max_distance: float = DEFAULT_MAX_DISTANCE
    """Maximum scaled centroid distance for a frame-to-frame link."""

    allow_division: bool = True
    """Whether to attach a second daughter to a matched parent (``1 -> 2``)."""

    division_distance: float = DEFAULT_MAX_DISTANCE
    """Maximum parent->second-daughter distance when ``allow_division`` is set."""


def _assign(
    src: np.ndarray, dst: np.ndarray, scale: np.ndarray, max_distance: float
) -> list[tuple[int, int]]:
    """Optimal one-to-one assignment of ``src`` rows to ``dst`` rows within range.

    Returns a list of ``(src_index, dst_index)`` pairs whose scaled distance is
    ``<= max_distance``. Costs above the threshold are masked to a large finite
    value so the assignment never prefers an out-of-range pair over an in-range
    one, then filtered out after solving.
    """
    if len(src) == 0 or len(dst) == 0:
        return []
    diff = (src[:, None, :] - dst[None, :, :]) * scale
    dist = np.sqrt((diff**2).sum(axis=2))
    big = max_distance * 1000.0 + dist.max() + 1.0
    cost = np.where(dist <= max_distance, dist, big)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if dist[r, c] <= max_distance]


def link_centroids(
    detections: dict[int, np.ndarray],
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    params: LinkParams | None = None,
) -> TrackingGraph:
    """Link per-timepoint centroids into a :class:`TrackingGraph`.

    ``detections`` maps ``timepoint -> (N, 3)`` centroid arrays in **voxel**
    coordinates. Node ids are assigned densely and deterministically (timepoint
    order, then detection order within a timepoint).
    """
    if params is None:
        params = LinkParams()
    scale_arr = np.asarray(scale, dtype=float)

    graph = TrackingGraph()
    ids_by_t: dict[int, list[int]] = {}
    next_id = 0
    for t in sorted(detections):
        ids: list[int] = []
        for z, y, x in detections[t]:
            graph.add_node(next_id, float(t), float(z), float(y), float(x))
            ids.append(next_id)
            next_id += 1
        ids_by_t[t] = ids

    times = sorted(detections)
    for t_a, t_b in zip(times, times[1:]):
        if t_b != t_a + 1:
            continue  # only link consecutive timepoints
        src = detections[t_a]
        dst = detections[t_b]
        pairs = _assign(src, dst, scale_arr, params.max_distance)
        matched_src = {i for i, _ in pairs}
        matched_dst = {j for _, j in pairs}
        for i, j in pairs:
            graph.add_edge(ids_by_t[t_a][i], ids_by_t[t_b][j])

        if not params.allow_division or len(src) == 0:
            continue
        # Attach each leftover t+1 detection to its nearest matched parent as a
        # second daughter, if within division_distance.
        parent_pos = src * scale_arr
        for j in range(len(dst)):
            if j in matched_dst:
                continue
            d = np.sqrt(((parent_pos - dst[j] * scale_arr) ** 2).sum(axis=1))
            order = np.argsort(d)
            for i in order:
                if d[i] > params.division_distance:
                    break
                if i in matched_src and graph.out_degree(ids_by_t[t_a][i]) < 2:
                    graph.add_edge(ids_by_t[t_a][i], ids_by_t[t_b][j])
                    matched_dst.add(j)
                    break
    return graph
