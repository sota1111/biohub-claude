"""End-to-end detection + linking pipeline for one OME-NGFF video.

Reads an OME-NGFF ``*.zarr`` volume ``(T, Z, Y, X)``, detects centroids per
timepoint (:mod:`biohub_tracking.detect`) and links them into a tracking graph
(:mod:`biohub_tracking.link`). The result is a
:class:`~biohub_tracking.graph.TrackingGraph` ready to be scored locally or
written to a submission CSV.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from .detect import (
    DEFAULT_SCALE,
    DetectParams,
    candidate_pool,
    detect_volume_series,
    detect_volume_series_with_descriptors,
)
from .graph import TrackingGraph
from .link import LinkParams, _component_sizes, link_centroids


def _open_image_array(zarr_path: Path | str):
    """Open the level-0 array of an OME-NGFF ``*.zarr`` group (falls back to array).

    Uses the ``zarr`` library when it is importable, but transparently falls back
    to the dependency-light :mod:`biohub_tracking.ngff` reader when it is not — a
    Kaggle Code-competition kernel image may not ship ``zarr`` and runs with no
    internet (SOT-1984). The fallback reads byte-identical volumes, so the
    champion produces the same submission either way.
    """
    try:
        import zarr
    except ImportError:
        from .ngff import open_ome_ngff_array

        return open_ome_ngff_array(zarr_path)

    root = zarr.open(str(zarr_path), mode="r")
    if hasattr(root, "shape") and not hasattr(root, "keys"):
        return root  # already an array
    # OME multiscales: level "0" is full resolution.
    return root["0"]


def _hypothesis_select_detections(
    arr,
    detect_params: DetectParams,
    link_params: LinkParams | None,
    scale: tuple[float, float, float],
    max_t: int | None,
) -> dict[int, np.ndarray]:
    """Ultrack-style multi-hypothesis detection selection by temporal support (SOT-2884).

    Builds a threshold-ladder candidate **pool** per frame
    (:func:`biohub_tracking.detect.candidate_pool` — local maxima above the lower
    robust-z gate, a superset of the champion peaks), links the whole pool with the
    champion motion cost but **without** short-track pruning / gap-recovery / the
    division overlay (so each candidate's weakly-connected-component size equals its
    *temporal support* — the length of the track it links into), and keeps only
    candidates whose support is ``>= detect_params.hypothesis_min_track``. Returns the
    surviving disjoint detections as ``{t: (M, 3)}`` for the caller to re-link with
    the champion linker. Deterministic; numpy/scipy-only.
    """
    n_t = arr.shape[0] if max_t is None else min(max_t, arr.shape[0])
    pool: dict[int, np.ndarray] = {}
    for t in range(n_t):
        coords, _strength = candidate_pool(arr[t], detect_params)
        pool[t] = coords

    # Support linking: the champion motion cost, but every candidate kept (no
    # min_track_length prune), no gap recovery and no division overlay, so a node's
    # component size is exactly the length of the motion-consistent track it earns.
    base_link = link_params if link_params is not None else LinkParams()
    support_params = dataclasses.replace(
        base_link,
        min_track_length=1,
        gap_recover=False,
        division_overlay=None,
    )
    support_graph = link_centroids(pool, scale=scale, params=support_params)
    sizes = _component_sizes(support_graph)

    min_track = int(detect_params.hypothesis_min_track)
    selected: dict[int, list[tuple[float, float, float]]] = {t: [] for t in range(n_t)}
    for node_id, (t, z, y, x) in support_graph.coords.items():
        if sizes.get(node_id, 1) >= min_track:
            selected[int(t)].append((z, y, x))
    return {
        t: np.asarray(rows, dtype=float).reshape(-1, 3)
        for t, rows in selected.items()
    }


def run_pipeline(
    zarr_path: Path | str,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    detect_params: DetectParams | None = None,
    link_params: LinkParams | None = None,
    max_t: int | None = None,
    learned_detector=None,
) -> TrackingGraph:
    """Detect and link one video into a tracking graph.

    ``learned_detector`` (SOT-2847) is the default-off receptacle for a torch
    learned detector: when a :class:`biohub_tracking.learned_detect.LearnedDetector`
    is passed, detection is routed through it (same ``detect_series`` contract as
    the classical detector) and the classical detect stage is skipped. ``None`` —
    the only value the champion ever supplies — takes the classical path below
    byte-for-byte unchanged.
    """
    arr = _open_image_array(zarr_path)
    # Learned-detector path (SOT-2847, off by default): a plugged-in learned
    # detector produces the same {t: (N, 3)} centroids the classical detector
    # would, so the linker/scorer/submission stages are untouched.
    if learned_detector is not None:
        detections = learned_detector.detect_series(arr, max_t=max_t)
        return link_centroids(detections, scale=scale, params=link_params)
    # Ultrack multi-hypothesis detection selection by temporal support (SOT-2884,
    # off by default): a whole-series detection re-anchor that produces the same
    # {t: (N, 3)} centroid contract, so the linker/submission stages are untouched.
    # Absent (detect_hypothesis_select=False) → the classical detect path below.
    if detect_params is not None and detect_params.detect_hypothesis_select:
        detections = _hypothesis_select_detections(
            arr, detect_params, link_params, scale, max_t
        )
        return link_centroids(detections, scale=scale, params=link_params)
    # Local appearance descriptors are only needed (and only computed) when the
    # linker's appearance term is active (SOT-2829); the distance-only champion
    # (appearance_weight == 0) takes the unchanged detect-only path byte-for-byte.
    if link_params is not None and link_params.appearance_weight > 0.0:
        detections, descriptors = detect_volume_series_with_descriptors(
            arr, detect_params, max_t=max_t, scale=scale
        )
        return link_centroids(
            detections, scale=scale, params=link_params, descriptors=descriptors
        )
    detections = detect_volume_series(arr, detect_params, max_t=max_t)
    return link_centroids(detections, scale=scale, params=link_params)
