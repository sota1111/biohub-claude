"""GT-free per-sequence density-regime covariates + leakage audit (SOT-2921).

Why this exists — the *family-mix wall*. The biohub holdout mixes a **dense**
lineage (``6bba``) with a **sparse / clean** lineage (``44b6``), and the two
require opposite operating points: a link/detect config tuned for one regresses
the other, so no single global config clears the 4/4 per-dataset non-regression
gate (SOT-2871/2882/2884/2909/2911/2920 all rejected on exactly this wall). The
cycle-5 reformulation (SOT-2919) is to make the operating point **regime
conditional** instead of global — but that is only sound if the regime can be
decided at *inference* time from the test input alone, with **no ground truth
and no family label**. Otherwise picking a config per regime is just fitting the
holdout labels (leakage), and any downstream promotion would be invalid.

This module builds that regime signal. It derives a **per-sequence density /
crowding covariate** from the detection point cloud the champion detector already
produces — the set of centroids ``(t, z, y, x)`` — using only pure ``numpy`` /
``scipy``. Every covariate here is:

* **GT-free** — computed from detection centroids only; no ``.geff`` labels, no
  :class:`~biohub_tracking.graph.TrackingGraph` ground truth ever enter.
* **test-computable** — the exact same function runs on a test ``*.zarr`` whose
  labels are hidden, because it consumes only the detector output.
* **label-independent** — a sequence's covariate does not depend on the
  held-out family's identity; the family prefix is used *only* to score
  separation after the fact, never to compute the covariate.

The leak-free guarantees are pinned by :mod:`tests.test_regime_covariate`.

Design notes
------------
* Covariates are computed **per frame then median-aggregated** across frames, so
  a few crowded or empty frames cannot dominate the sequence summary.
* Distances are reported in **physical microns** using the fixed acquisition
  scale :data:`ACQUISITION_SCALE` (microns/voxel, identical across all
  sequences), so anisotropic voxels do not bias spacing. The scale is a global
  affine constant — it cannot manufacture separation, it only makes the numbers
  physically interpretable.
* The **primary** covariate is the median nearest-neighbour spacing
  (``median_knn_um``): the classic Cell-Tracking-Challenge "observed cell
  spacing → dataset-specific parameters" signal (GT-free). Density and local
  crowding are recorded as auxiliaries.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np

from .cv import FamilyResult, aggregate

# Acquisition scale (microns / voxel), Z, Y, X. Mirrors
# :data:`biohub_tracking.detect.DEFAULT_SCALE`; kept as a local constant so this
# module (and its unit tests) stay dependency-light and never import the heavy
# detection stack. ``tests.test_regime_covariate`` asserts it never drifts from
# the detector's constant. It is a fixed acquisition property — same for every
# sequence — so it is fully test-computable and cannot create separation.
ACQUISITION_SCALE: tuple[float, float, float] = (1.625, 0.40625, 0.40625)

# Physical radius (microns) defining "local" for the crowding covariate: the
# median number of other detections within this radius of a cell.
DEFAULT_CROWDING_RADIUS_UM: float = 15.0


def _frames_as_um(
    detections: Mapping[int, np.ndarray],
    scale: tuple[float, float, float],
) -> list[np.ndarray]:
    """Convert ``{t: (N,3) voxel (z,y,x)}`` detections to per-frame micron arrays.

    Returns one ``(N, 3)`` float array per frame (in physical microns), ordered
    by timepoint for determinism. Frames with no detections yield an empty
    ``(0, 3)`` array and are kept so counts stay faithful.
    """
    scale_arr = np.asarray(scale, dtype=float)
    frames: list[np.ndarray] = []
    for t in sorted(detections):
        pts = np.asarray(detections[t], dtype=float).reshape(-1, 3)
        frames.append(pts * scale_arr)
    return frames


def sequence_covariates(
    detections: Mapping[int, np.ndarray],
    scale: tuple[float, float, float] = ACQUISITION_SCALE,
    *,
    crowding_radius_um: float = DEFAULT_CROWDING_RADIUS_UM,
    min_cells_per_frame: int = 2,
) -> dict:
    """Compute GT-free per-sequence density/crowding covariates.

    Parameters
    ----------
    detections
        ``{t: (N, 3)}`` voxel centroids ``(z, y, x)`` — the detector output for
        one sequence. This is the *only* data the covariate depends on.
    scale
        Microns/voxel ``(z, y, x)``; defaults to the fixed acquisition constant.
    crowding_radius_um
        Radius (microns) for the local-crowding covariate.
    min_cells_per_frame
        Frames with fewer detections than this are skipped for the spacing /
        crowding covariates (a nearest-neighbour distance needs >= 2 points).

    Returns
    -------
    dict
        Deterministic scalars:

        * ``median_knn_um`` (**primary**) — median over frames of the median
          nearest-neighbour distance among same-frame cells (cell spacing;
          *large ⇒ sparse*, *small ⇒ dense*).
        * ``density_cells_per_1e6_um3`` — median over frames of cells per
          bounding-box volume (*large ⇒ dense*).
        * ``local_crowding`` — median over frames of the median number of other
          cells within ``crowding_radius_um`` of a cell (*large ⇒ dense*).
        * ``median_cells_per_frame`` — median detection count per frame.
        * bookkeeping: ``n_frames_total``, ``n_frames_scored``,
          ``n_detections_total``, ``crowding_radius_um``.

    Pure, deterministic, and label-free. NaN is returned for a covariate that has
    no scorable frame (e.g. every frame under ``min_cells_per_frame``).
    """
    from scipy.spatial import cKDTree

    frames = _frames_as_um(detections, scale)
    n_frames_total = len(frames)
    n_detections_total = int(sum(f.shape[0] for f in frames))
    counts = [f.shape[0] for f in frames]

    knn_per_frame: list[float] = []
    density_per_frame: list[float] = []
    crowding_per_frame: list[float] = []

    for pts in frames:
        n = pts.shape[0]
        if n < max(2, min_cells_per_frame):
            continue
        tree = cKDTree(pts)
        # k=2: nearest neighbour excluding self (column 0 is the point itself).
        dists, _ = tree.query(pts, k=2)
        nn = dists[:, 1]
        knn_per_frame.append(float(np.median(nn)))

        # Local crowding: neighbours strictly within the radius, minus self.
        neigh_counts = tree.query_ball_point(pts, r=float(crowding_radius_um))
        crowd = np.array([len(c) - 1 for c in neigh_counts], dtype=float)
        crowding_per_frame.append(float(np.median(crowd)))

        # Density: cells per physical bounding-box volume. Guard degenerate axes
        # (a flat extent along an axis) with a >=1 voxel floor so volume is finite
        # and positive; a single-cell frame is already excluded above.
        extent = pts.max(axis=0) - pts.min(axis=0)
        floor = np.asarray(scale, dtype=float)
        extent = np.maximum(extent, floor)
        volume_um3 = float(np.prod(extent))
        if volume_um3 > 0:
            density_per_frame.append(n / volume_um3 * 1.0e6)

    def _median(vals: list[float]) -> float:
        return float(np.median(vals)) if vals else float("nan")

    return {
        "median_knn_um": _median(knn_per_frame),
        "density_cells_per_1e6_um3": _median(density_per_frame),
        "local_crowding": _median(crowding_per_frame),
        "median_cells_per_frame": (
            float(np.median(counts)) if counts else float("nan")
        ),
        "crowding_radius_um": float(crowding_radius_um),
        "n_frames_total": n_frames_total,
        "n_frames_scored": len(knn_per_frame),
        "n_detections_total": n_detections_total,
    }


def assign_regime(
    covariate_value: float,
    threshold: float,
    *,
    dense_is_low: bool = True,
) -> str:
    """Label a sequence ``"dense"`` / ``"sparse"`` from one observable covariate.

    ``dense_is_low`` is ``True`` for spacing-type covariates (``median_knn_um``:
    tighter spacing ⇒ denser) and ``False`` for count-type covariates
    (``density`` / ``crowding``: larger ⇒ denser). Ties resolve to ``"dense"``.
    """
    if math.isnan(covariate_value):
        return "unknown"
    if dense_is_low:
        return "dense" if covariate_value <= threshold else "sparse"
    return "dense" if covariate_value >= threshold else "sparse"


def separation_report(
    covariates_by_seq: Mapping[str, dict],
    group_of: Callable[[str], str],
    *,
    key: str = "median_knn_um",
    dense_is_low: bool = True,
) -> dict:
    """Quantify how well one covariate separates two observed groups (leak-free).

    ``group_of`` maps a sequence name to its true group label (e.g. the family
    prefix ``6bba`` / ``44b6``). The grouping is used **only to evaluate**
    separation — never to compute the covariate — so this is an honest audit of
    whether the GT-free covariate recovers the dense/sparse split.

    Returns per-group value ranges, whether the two groups' value intervals are
    disjoint (``separable``), the ``margin`` (gap between the nearest cross-group
    values; negative ⇒ overlap), a midpoint ``threshold``, and a per-sequence
    predicted-regime table plus whether the threshold classifies every sequence
    consistently with its group's dense/sparse side.
    """
    values = {
        name: float(cov.get(key, float("nan")))
        for name, cov in covariates_by_seq.items()
    }
    groups: dict[str, list[float]] = {}
    for name, val in values.items():
        if math.isnan(val):
            continue
        groups.setdefault(group_of(name), []).append(val)

    group_ranges = {
        g: {"min": min(vs), "max": max(vs), "mean": float(np.mean(vs)), "n": len(vs)}
        for g, vs in groups.items()
    }

    result: dict = {
        "covariate": key,
        "values": values,
        "groups": group_ranges,
    }

    if len(groups) != 2:
        result.update(
            {
                "separable": None,
                "margin": None,
                "threshold": None,
                "note": "separation is only defined for exactly two groups",
            }
        )
        return result

    (g_lo, vs_lo), (g_hi, vs_hi) = sorted(
        groups.items(), key=lambda kv: np.mean(kv[1])
    )
    lo_max, hi_min = max(vs_lo), min(vs_hi)
    # Margin between the group with lower values and the group with higher
    # values; positive ⇒ a gap (disjoint intervals), negative ⇒ overlap.
    margin = hi_min - lo_max
    separable = margin > 0
    threshold = (lo_max + hi_min) / 2.0

    # Which group is "dense" under this covariate's orientation.
    dense_group = g_lo if dense_is_low else g_hi
    predicted = {
        name: assign_regime(val, threshold, dense_is_low=dense_is_low)
        for name, val in values.items()
        if not math.isnan(val)
    }
    # A sequence is classified correctly iff its predicted regime matches the
    # side its true group sits on.
    correct = {
        name: (
            (pred == "dense") == (group_of(name) == dense_group)
        )
        for name, pred in predicted.items()
    }

    result.update(
        {
            "lower_group": g_lo,
            "higher_group": g_hi,
            "dense_group": dense_group,
            "margin": float(margin),
            "separable": bool(separable),
            "threshold": float(threshold),
            "predicted_regime": predicted,
            "classified_correctly": correct,
            "all_correct": all(correct.values()),
        }
    )
    return result


def regime_stratified_cv(
    rows: tuple[FamilyResult, ...] | list[FamilyResult],
    regime_by_family: Mapping[str, str],
) -> dict:
    """Regime-stratified CV report: per-family AND per-regime micro Jaccards.

    Re-aggregates already-scored per-family CV rows (:class:`FamilyResult`) into
    per-regime buckets using an **observable** regime label per family (from
    :func:`assign_regime`), so the family-mix wall is made visible: the dense and
    sparse regimes report their own ``micro_adj`` / ``micro_raw`` edge Jaccards
    instead of being blended into one 6bba-dominated micro.

    Pure function of the CV rows + the regime labelling (no disk, no GT). Reuses
    :func:`biohub_tracking.eval.cv.aggregate` so the per-regime micro arithmetic
    is byte-identical to the global CV.
    """
    rows = tuple(rows)
    per_family = {
        r.name: {
            "regime": regime_by_family.get(r.name, "unknown"),
            "lineage": r.lineage,
            "adj_edge_jaccard": round(r.adj_edge_jaccard, 4),
            "edge_jaccard": round(r.edge_jaccard, 4),
            "weight": r.weight,
        }
        for r in rows
    }

    by_regime: dict[str, list[FamilyResult]] = {}
    for r in rows:
        by_regime.setdefault(regime_by_family.get(r.name, "unknown"), []).append(r)

    total_weight = sum(r.weight for r in rows)
    per_regime = {}
    for regime, regime_rows in sorted(by_regime.items()):
        agg = aggregate(regime_rows)
        weight = sum(r.weight for r in regime_rows)
        per_regime[regime] = {
            "micro_adj_edge_jaccard": round(agg.micro_adj_edge_jaccard, 4),
            "micro_edge_jaccard": round(agg.micro_edge_jaccard, 4),
            "families": [r.name for r in regime_rows],
            "weight": weight,
            "weight_share": (
                round(weight / total_weight, 4) if total_weight > 0 else None
            ),
        }

    return {"per_family": per_family, "per_regime": per_regime}
