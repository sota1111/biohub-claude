"""End-to-end detection + linking pipeline for one OME-NGFF video.

Reads an OME-NGFF ``*.zarr`` volume ``(T, Z, Y, X)``, detects centroids per
timepoint (:mod:`biohub_tracking.detect`) and links them into a tracking graph
(:mod:`biohub_tracking.link`). The result is a
:class:`~biohub_tracking.graph.TrackingGraph` ready to be scored locally or
written to a submission CSV.
"""

from __future__ import annotations

from pathlib import Path

from .detect import (
    DEFAULT_SCALE,
    DetectParams,
    detect_volume_series,
    detect_volume_series_with_descriptors,
)
from .graph import TrackingGraph
from .link import LinkParams, link_centroids


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
