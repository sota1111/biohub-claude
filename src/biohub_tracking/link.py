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
  matching the competition's division definition. Two **over-split suppressors**
  keep spurious forks in check on dense volumes (SOT-2762): ``division_distance``
  can be tightened below ``max_distance``, and ``division_max_sibling_ratio``
  gates the second daughter on being roughly balanced with the primary daughter
  (a real mitotic split yields two daughters symmetric about the parent, so a
  detection much farther than the assigned child is not a plausible sibling).

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
    """Maximum parent->second-daughter distance when ``allow_division`` is set.

    An over-split suppressor: tightening this below :attr:`max_distance` restricts
    divisions to close, high-confidence sibling pairs, so a dense volume does not
    spray spurious forks (each of which is a division FP and, when its extra edge
    is evaluable, an edge FP too)."""

    division_max_sibling_ratio: float = 0.0
    """Sibling-balance over-split gate for divisions (SOT-2762).

    A real mitotic split produces two daughters roughly equidistant from the
    parent's last position. When ``> 0``, a leftover ``t+1`` detection is attached
    as a parent's second daughter only if its scaled parent-distance ``d2`` is
    ``<= division_max_sibling_ratio * d1``, where ``d1`` is the scaled distance
    from that parent to its *primary* (one-to-one assigned) daughter. This rejects
    attaching a distant unrelated detection as a fake sibling, the dominant
    over-split failure mode on the dense ``6bba`` families. ``0.0`` disables the
    gate (any leftover within ``division_distance`` may divide — the pre-SOT-2762
    behaviour, byte-for-byte)."""

    velocity_gain: float = 0.0
    """Constant-velocity motion prediction for the assignment (SOT-2369).

    A cell that already has an incoming edge (``t-1 -> t``) carries a velocity
    ``v = pos(t) - pos(t-1)``. When matching ``t -> t+1`` its predicted position
    becomes ``pos(t) + velocity_gain * v`` — a **damped** constant-velocity
    extrapolation ported from the public V40 lineage tracker's motion-aware
    reassignment (``q_hat = q + 0.5*(q - q_pred)``). The assignment then ranks
    candidate successors by distance to the *predicted* position, so a moving
    cell prefers the detection it is drifting toward rather than the closest
    stationary one. ``0.0`` disables prediction and reproduces the memoryless
    nearest-neighbour champion **byte-for-byte**."""

    velocity_disp_weight: float = 0.05
    """Tie-breaker weight on raw (unpredicted) displacement in the motion cost.

    Mirrors the ``0.05 * ||q_j - q_i||`` regulariser in the reference cost, so
    among motion-consistent candidates the assignment still slightly prefers the
    physically nearer detection. Only used when ``velocity_gain != 0``."""

    motion_gate_on_prediction: bool = False
    """Admissibility gate when motion prediction is active.

    ``False`` (default) keeps the feasibility gate on the **actual** scaled
    distance ``||pos(t) - pos(t+1)|| <= max_distance`` — motion only *re-ranks*
    within the champion's existing feasible set, so it can never introduce a new
    long-range edge (pure, low-risk improvement). ``True`` gates on the predicted
    distance instead, letting a fast, motion-consistent cell link slightly beyond
    ``max_distance`` in raw terms."""

    min_track_length: int = 1
    """Drop weakly-connected track fragments with fewer than this many nodes
    (SOT-2369, ported from the reference tracker's ``FILTER_SHORT_TRACKS``).

    A real cell persists across many frames, so a detection that fails to link
    into a multi-frame track is almost always noise. After linking, each track is
    a weakly-connected component; components with ``< min_track_length`` nodes are
    removed entirely (their nodes and edges). This is decoupled from detection
    thresholding: it prunes spurious detections *by tracking topology* rather than
    by intensity. Because a light-sheet volume is annotated sparsely, an
    over-predicting champion pays a node-count penalty
    ``J_adj = J·(1 - 0.1·(N_pred - N_true)/N_true)``; dropping isolated singletons
    (``min_track_length=2``) removes nodes that carry **no** matched edge, so edge
    TP/FP/FN are untouched while ``N_pred`` falls — a strict adjusted-Jaccard gain
    whenever the pipeline over-predicts. ``1`` keeps every node (champion
    behaviour, byte-for-byte)."""


def _prune_short_tracks(graph: TrackingGraph, min_nodes: int) -> TrackingGraph:
    """Drop weakly-connected components smaller than ``min_nodes`` nodes."""
    if min_nodes <= 1:
        return graph
    parent: dict[int, int] = {n: n for n in graph.node_ids()}

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:  # path compression
            parent[a], a = root, parent[a]
        return root

    for src, dst in graph.edges:
        parent[find(src)] = find(dst)

    sizes: dict[int, int] = {}
    for n in graph.node_ids():
        r = find(n)
        sizes[r] = sizes.get(r, 0) + 1
    keep = [n for n in graph.node_ids() if sizes[find(n)] >= min_nodes]
    return graph.subgraph(keep)


def _assign(
    src: np.ndarray, dst: np.ndarray, scale: np.ndarray, max_distance: float,
    src_pred: np.ndarray | None = None, disp_weight: float = 0.0,
    gate_on_prediction: bool = False,
) -> list[tuple[int, int]]:
    """Optimal one-to-one assignment of ``src`` rows to ``dst`` rows within range.

    Returns a list of ``(src_index, dst_index)`` pairs whose scaled distance is
    ``<= max_distance``. Costs above the threshold are masked to a large finite
    value so the assignment never prefers an out-of-range pair over an in-range
    one, then filtered out after solving.

    When ``src_pred`` is given (constant-velocity predicted source positions),
    the assignment *cost* is the scaled distance from the **predicted** source to
    the destination plus ``disp_weight`` times the raw (unpredicted) scaled
    distance; the feasibility gate uses the predicted distance if
    ``gate_on_prediction`` else the actual distance.
    """
    if len(src) == 0 or len(dst) == 0:
        return []
    diff = (src[:, None, :] - dst[None, :, :]) * scale
    dist = np.sqrt((diff**2).sum(axis=2))
    if src_pred is None:
        gate = dist
        cost_base = dist
    else:
        diff_pred = (src_pred[:, None, :] - dst[None, :, :]) * scale
        dist_pred = np.sqrt((diff_pred**2).sum(axis=2))
        gate = dist_pred if gate_on_prediction else dist
        cost_base = dist_pred + disp_weight * dist
    big = max_distance * 1000.0 + cost_base.max() + 1.0
    cost = np.where(gate <= max_distance, cost_base, big)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if gate[r, c] <= max_distance]


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

    # Per-node incoming displacement, indexed by (timepoint, detection-index), so
    # a matched cell can predict where it is drifting next frame. Populated as
    # links are made; empty when ``velocity_gain == 0`` (memoryless champion path).
    times = sorted(detections)
    velocity_by_index: dict[tuple[int, int], np.ndarray] = {}
    for t_a, t_b in zip(times, times[1:]):
        if t_b != t_a + 1:
            continue  # only link consecutive timepoints
        src = detections[t_a]
        dst = detections[t_b]
        if params.velocity_gain and len(src):
            src_pred = np.array(
                [
                    src[i] + params.velocity_gain * velocity_by_index.get((t_a, i), 0.0)
                    for i in range(len(src))
                ],
                dtype=float,
            )
            pairs = _assign(
                src, dst, scale_arr, params.max_distance,
                src_pred=src_pred, disp_weight=params.velocity_disp_weight,
                gate_on_prediction=params.motion_gate_on_prediction,
            )
        else:
            pairs = _assign(src, dst, scale_arr, params.max_distance)
        # Record each child's incoming displacement so it can predict t+1 -> t+2.
        if params.velocity_gain:
            for i, j in pairs:
                velocity_by_index[(t_b, j)] = dst[j] - src[i]
        matched_src = {i for i, _ in pairs}
        matched_dst = {j for _, j in pairs}
        for i, j in pairs:
            graph.add_edge(ids_by_t[t_a][i], ids_by_t[t_b][j])

        if not params.allow_division or len(src) == 0:
            continue
        # Scaled distance from each matched parent to its primary (assigned)
        # daughter, for the sibling-balance over-split gate.
        primary_dist: dict[int, float] = {}
        if params.division_max_sibling_ratio > 0.0:
            for i, j in pairs:
                primary_dist[i] = float(
                    np.sqrt((((src[i] - dst[j]) * scale_arr) ** 2).sum())
                )
        # Attach each leftover t+1 detection to its nearest matched parent as a
        # second daughter, if within division_distance (and, when the sibling
        # ratio gate is on, balanced against that parent's primary daughter).
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
                    if params.division_max_sibling_ratio > 0.0 and (
                        d[i] > params.division_max_sibling_ratio * primary_dist.get(i, 0.0)
                    ):
                        continue  # sibling too far vs primary daughter → not a split
                    graph.add_edge(ids_by_t[t_a][i], ids_by_t[t_b][j])
                    matched_dst.add(j)
                    break

    if params.min_track_length > 1:
        graph = _prune_short_tracks(graph, params.min_track_length)
    return graph
